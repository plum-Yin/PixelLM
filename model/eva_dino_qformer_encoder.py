import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from types import SimpleNamespace
from transformers import CLIPVisionModel, Blip2QFormerModel, CLIPImageProcessor


class OptionalReNorm(nn.Module):
    """
    将 CLIP 范围 [-1,1] 的图像近似还原到 [0,1]，再做 ImageNet 规范化，供 DINO 使用。
    """
    def __init__(self, enable=True):
        super().__init__()
        self.enable = enable
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x_clip_norm):
        if not self.enable:
            return x_clip_norm
        x01 = (x_clip_norm + 1.0) * 0.5
        return (x01 - self.mean) / self.std


class DINOv2TimmWrapper(nn.Module):
    """
    通过 timm 加载 DINOv2，并返回 [B, N, D] 的 patch tokens
    """
    def __init__(self, model_name="vit_large_patch14_dinov2.lvd142m"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.embed_dim = self.model.embed_dim  # 例如 vit_large_patch14_dinov2 = 1024

    def forward(self, images):
        # timm 的 ViT forward_features 返回字典，x 为 [B, N+1, D] (含 CLS)
        feats = self.model.forward_features(images)
        if isinstance(feats, dict) and "x" in feats:
            tokens = feats["x"]  # [B, N+1, D]
        else:
            tokens = feats
        # 去掉 CLS token
        patch_tokens = tokens[:, 1:, :]  # [B, N, D]
        return SimpleNamespace(last_hidden_state=patch_tokens)


class EvaDinoQFormerVisionTower(nn.Module):
    """
    自定义 VisionTower：EVA-CLIP ⊕ DINOv2，经 Cross-Attn 融合为密集 token（I_v），
    同时支持可选 Q-Former 短序列压缩。为对接 LLaVA/PixelLM 管线：
      - forward(images) 返回 (image_features=I_v, pre_image_features=[I_clip])
      - 提供 image_processor / hidden_size / dtype / device / dummy_feature 等属性
      - 支持 resize_vision_tower 以 448 等尺寸运行 EVA-CLIP
    构造签名与 CLIPVisionTower 对齐：__init__(vision_tower, args, delay_load=False)
    """

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.args = args

        # 训练配置
        self.pad_vit = getattr(args, "pad_train_clip_images", False)
        self.resize_vision_tower = getattr(args, "resize_vision_tower", False)
        self.resize_vision_tower_size = getattr(args, "resize_vision_tower_size", 224)

        # 预处理器
        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        except Exception:
            self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # 模块占位
        self.eva_clip = None
        self.dino = None
        self.dino_proj = None
        self.cross_attention = None
        self.qformer = None
        self.q_proj_in = None
        self.q_proj_out = None
        self.learnable_query = None
        self._clip_dim = None

        if not delay_load:
            self.load_model()

    def load_model(self):
        # 1) EVA-CLIP
        # 若用户提供 eva_clip_path 优先使用，否则用 vision_tower_name
        eva_name = getattr(self.args, 'eva_clip_path', None) or self.vision_tower_name
        self.eva_clip = CLIPVisionModel.from_pretrained(eva_name)
        clip_dim = self.eva_clip.config.hidden_size
        self._clip_dim = clip_dim

        # 支持可选位置编码 resize 到更大输入
        if self.resize_vision_tower:
            vt = self.eva_clip
            origin_p_num = int(vt.vision_model.embeddings.num_patches ** 0.5)
            vision_tower_embed_dim = vt.vision_model.embeddings.embed_dim
            vt.vision_model.embeddings.image_size = self.resize_vision_tower_size
            vt.vision_model.embeddings.num_patches = (
                self.resize_vision_tower_size // vt.vision_model.embeddings.patch_size
            ) ** 2
            vt.vision_model.embeddings.num_positions = vt.vision_model.embeddings.num_patches + 1
            vt.vision_model.embeddings.register_buffer(
                "position_ids", torch.arange(vt.vision_model.embeddings.num_positions).expand((1, -1))
            )
            new_p_num = int(vt.vision_model.embeddings.num_patches ** 0.5)

            origin_position_embedding_weight = vt.vision_model.embeddings.position_embedding.weight
            origin_position_embedding_weight_cls = origin_position_embedding_weight[-1:]
            origin_position_embedding_weight = (
                origin_position_embedding_weight[:-1]
                .permute(1, 0)
                .view(1, vision_tower_embed_dim, origin_p_num, origin_p_num)
            )
            new_position_embedding_weight = F.interpolate(
                origin_position_embedding_weight,
                (new_p_num, new_p_num),
                mode="bilinear",
                align_corners=False,
            )[0]
            new_position_embedding_weight = new_position_embedding_weight.flatten(-2).permute(1, 0)
            new_position_embedding_weight = torch.cat(
                (new_position_embedding_weight, origin_position_embedding_weight_cls), dim=0
            )
            vt.vision_model.embeddings.position_embedding = nn.Embedding(
                vt.vision_model.embeddings.num_positions, vision_tower_embed_dim
            )
            vt.vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(
                new_position_embedding_weight
            ).to(origin_position_embedding_weight)
            vt.vision_model.embeddings.position_ids = (
                vt.vision_model.embeddings.position_ids.to(origin_position_embedding_weight.device)
            )

        # 2) DINOv2（默认 timm）
        dino_model_name = getattr(self.args, "dino_model_name", "vit_large_patch14_dinov2.lvd142m")
        self.dino = DINOv2TimmWrapper(dino_model_name)
        dino_dim = self.dino.embed_dim
        self.imagenet_norm = OptionalReNorm(enable=True)

        # 3) 维度对齐 + Cross Attention
        self.dino_proj = nn.Linear(dino_dim, clip_dim, bias=False)
        num_heads = getattr(self.args, "cross_attn_heads", 8)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=clip_dim, num_heads=num_heads, batch_first=True
        )

        # 4) Q-Former（可选）
        qformer_path = getattr(self.args, "qformer_path", None)
        if qformer_path:
            self.qformer = Blip2QFormerModel.from_pretrained(qformer_path)
            qformer_dim = self.qformer.config.hidden_size
            if qformer_dim != clip_dim:
                self.q_proj_in = nn.Linear(clip_dim, qformer_dim, bias=False)
                self.q_proj_out = nn.Linear(qformer_dim, clip_dim, bias=False)
            else:
                self.q_proj_in = nn.Identity()
                self.q_proj_out = nn.Identity()
            num_query = getattr(self.args, "num_query", 32)
            self.learnable_query = nn.Parameter(torch.randn(num_query, clip_dim))

        self.is_loaded = True
        # 默认冻结视觉模型，后续由上层按需控制 requires_grad
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images, attention_mask=None):
        assert self.is_loaded, "Vision tower not loaded; call load_model() first."
        # EVA-CLIP：去 CLS
        I_clip = self.eva_clip(images, output_hidden_states=False).last_hidden_state[:, 1:, :]
        # DINO：ImageNet 归一化
        I_dino_in = self.imagenet_norm(images)
        I_dino = self.dino_proj(self.dino(I_dino_in).last_hidden_state)

        # Cross-Attention 融合
        I_v, _ = self.cross_attention(I_clip, I_dino, I_dino)
        image_features = I_v.to(images.dtype)
        pre_image_features = [I_clip.to(images.dtype)]
        return image_features, pre_image_features

    @torch.no_grad()
    def forward_qformer(self, images, text_embeds=None):
        if self.qformer is None:
            raise RuntimeError("Q-Former not initialized; provide qformer_path in args.")
        I_clip = self.eva_clip(images, output_hidden_states=False).last_hidden_state[:, 1:, :]
        I_dino_in = self.imagenet_norm(images)
        I_dino = self.dino_proj(self.dino(I_dino_in).last_hidden_state)
        I_v, _ = self.cross_attention(I_clip, I_dino, I_dino)

        B = images.size(0)
        learnable_q = self.learnable_query.unsqueeze(0).expand(B, -1, -1)
        q_input = torch.cat([learnable_q, text_embeds], dim=1) if text_embeds is not None else learnable_q
        q_input = self.q_proj_in(q_input)
        qf_out = self.qformer(query_embeds=q_input, encoder_hidden_states=I_v).last_hidden_state
        img_embeds = qf_out[:, :self.learnable_query.size(0), :]
        img_embeds = self.q_proj_out(img_embeds).to(images.dtype)
        return img_embeds

    @property
    def dtype(self):
        return self.eva_clip.dtype if self.eva_clip is not None else torch.float16

    @property
    def device(self):
        return self.eva_clip.device if self.eva_clip is not None else torch.device("cpu")

    @property
    def config(self):
        return self.eva_clip.config if self.eva_clip is not None else SimpleNamespace(hidden_size=self._clip_dim or 1024)

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
