import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from transformers import (
    CLIPVisionModel,
    Blip2QFormerModel,
    CLIPImageProcessor,
)
try:
    # Available in many Transformers versions; used if present
    from transformers import Dinov2Model  # type: ignore
    _HAS_DINOV2 = True
except Exception:
    Dinov2Model = None  # type: ignore
    _HAS_DINOV2 = False
try:
    # ViT fallback if Dinov2 is unavailable
    from transformers import ViTModel  # type: ignore
    _HAS_VIT = True
except Exception:
    ViTModel = None  # type: ignore
    _HAS_VIT = False


class OptionalReNorm(nn.Module):
    """
    将 CLIP 预处理后的像素（均值/方差归一化）还原到 [0,1]，再做 ImageNet 规范化，供 DINO/ViT 使用。
    注意：输入应为 CLIPImageProcessor.preprocess 的 `pixel_values`（CHW，float，已归一化）。
    """
    def __init__(self, enable: bool = True, clip_mean: torch.Tensor = None, clip_std: torch.Tensor = None):
        super().__init__()
        self.enable = enable
        # ImageNet/DINO 标准化参数（[0,1] 空间）
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('imagenet_mean', imagenet_mean)
        self.register_buffer('imagenet_std', imagenet_std)

        # CLIP 预处理参数（[0,1] 空间），用于逆变换回 [0,1]
        if clip_mean is None or clip_std is None:
            # 若未提供，则退化为直传（不推荐，但保证健壮性）
            clip_mean = torch.zeros(1, 3, 1, 1)
            clip_std = torch.ones(1, 3, 1, 1)
        self.register_buffer('clip_mean', clip_mean.view(1, 3, 1, 1))
        self.register_buffer('clip_std', clip_std.view(1, 3, 1, 1))

        # 记录是否已初始化
        self.initialized = True

    def forward(self, x_clip_norm: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return x_clip_norm
        # 逆 CLIP 归一化：x01 ∈ [0,1]
        x01 = x_clip_norm * self.clip_std + self.clip_mean
        # 再做 ImageNet 标准化
        return (x01 - self.imagenet_mean) / self.imagenet_std


class DINOv2HFWrapper(nn.Module):
    """
    使用 Hugging Face Transformers 加载 DINOv2；若不可用则回退到 ViT。
    - 返回对象包含 last_hidden_state（去除 CLS）: [B, N, D]
    - 仅依赖 transformers==4.31.0 生态，不引入 timm
    """
    def __init__(self, model_name_or_path: str = "facebook/dinov2-large"):
        super().__init__()
        self.backend = None
        self.embed_dim = None

        model = None
        # 1) 优先 DINOv2（若 transformers 版本支持，且权重可用）
        if _HAS_DINOV2 and Dinov2Model is not None:
            try:
                model = Dinov2Model.from_pretrained(model_name_or_path)
                self.backend = "dinov2"
            except Exception:
                model = None

        # 2) 回退 ViT（大概率 transformers 都可用）
        if model is None and _HAS_VIT and ViTModel is not None:
            try:
                # 合理的 ViT 大模型作为近似替代
                fallback = (
                    "google/vit-large-patch16-224-in21k"
                    if model_name_or_path is None
                    else model_name_or_path
                )
                # 若传入的是 timm 风格名称，直接忽略并使用 fallback
                if "dinov2" in str(model_name_or_path).lower() or \
                   "vit_large_patch" in str(model_name_or_path).lower():
                    fallback = "google/vit-large-patch16-224-in21k"
                model = ViTModel.from_pretrained(fallback)
                self.backend = "vit"
            except Exception:
                model = None

        if model is None:
            raise RuntimeError(
                "Failed to load vision backbone from transformers. Dinov2Model/ViTModel unavailable."
            )

        self.model = model
        self.embed_dim = getattr(self.model.config, "hidden_size")

    def forward(self, images: torch.Tensor):
        outputs = self.model(pixel_values=images)
        tokens = outputs.last_hidden_state  # [B, 1+N, D]
        patch_tokens = tokens[:, 1:, :]  # 去 CLS -> [B, N, D]
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
            print(f"Warning: failed to load image_processor for {self.vision_tower_name}, using openai/clip-vit-large-patch14 instead.")
            self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # 模块占位
        self.eva_clip = None
        self.dino = None
        self.dino_proj = None
        self.cross_attention = None
        self.qformer = None
        self.q_proj_in = None
        self.q_proj_out = None
        self.q_kv_proj = None
        self.learnable_query = None
        self._clip_dim = None
        self._qformer_debug_once = False

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

        # 2) DINOv2（transformers 实现）
        # 兼容外部传参名：优先使用 --dino_path；若未提供则使用 Dinov2 预设
        dino_model_name = getattr(self.args, "dino_path", None)
        if dino_model_name in (None, "",):
            dino_model_name = "facebook/dinov2-large"
        self.dino = DINOv2HFWrapper(dino_model_name)
        dino_dim = self.dino.embed_dim
        # 使用实际的 CLIP 均值/方差做逆归一化，再做 ImageNet 规范化
        try:
            clip_mean = torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1)
            clip_std = torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1)
        except Exception:
            clip_mean = torch.zeros(1, 3, 1, 1)
            clip_std = torch.ones(1, 3, 1, 1)
        self.imagenet_norm = OptionalReNorm(enable=True, clip_mean=clip_mean, clip_std=clip_std)

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
            # 对 encoder_hidden_states (I_v) 做 KV 侧维度对齐
            qformer_kv_dim = getattr(self.qformer.config, "encoder_hidden_size", qformer_dim)
            if qformer_kv_dim != clip_dim:
                self.q_kv_proj = nn.Linear(clip_dim, qformer_kv_dim, bias=False)
            else:
                self.q_kv_proj = nn.Identity()
            num_query = getattr(self.args, "num_query", 32)
            self.learnable_query = nn.Parameter(torch.randn(num_query, clip_dim))

        self.is_loaded = True
        # 默认冻结视觉模型，后续由上层按需控制 requires_grad
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images, attention_mask=None):
        assert self.is_loaded, "Vision tower not loaded; call load_model() first."
        # 若输入分辨率与 CLIP 位置编码不一致，则在前向时自适应到设定的 image_size
        images_clip_in = images
        vt = self.eva_clip
        try:
            expected = vt.vision_model.embeddings.image_size
        except Exception:
            expected = None
        if expected is not None:
            h, w = images.shape[-2:]
            if h != expected or w != expected:
                images_clip_in = F.interpolate(
                    images, size=(expected, expected), mode="bilinear", align_corners=False
                )

        # EVA-CLIP：去 CLS
        I_clip = vt(images_clip_in, output_hidden_states=False).last_hidden_state[:, 1:, :]
        # DINO：ImageNet 归一化
        I_dino_in = self.imagenet_norm(images)
        # 大多数 HF 视觉骨干使用 float32；为稳妥起见在此处转为 float32
        I_dino = self.dino_proj(self.dino(I_dino_in.float()).last_hidden_state)

        # Cross-Attention 融合
        I_v, _ = self.cross_attention(I_clip, I_dino, I_dino)
        image_features = I_v.to(images.dtype)
        pre_image_features = [I_clip.to(images.dtype)]
        return image_features, pre_image_features

    @torch.no_grad()
    def forward_qformer(self, images=None, text_embeds=None, image_features=None):
        if self.qformer is None:
            raise RuntimeError("Q-Former not initialized; provide qformer_path in args.")
        vt = self.eva_clip
        # Use cached dense image features if provided to avoid recomputing vision tower
        if image_features is not None:
            I_v = image_features
        else:
            assert images is not None, "forward_qformer requires either images or image_features"
            images_clip_in = images
            try:
                expected = vt.vision_model.embeddings.image_size
            except Exception:
                expected = None
            if expected is not None:
                h, w = images.shape[-2:]
                if h != expected or w != expected:
                    images_clip_in = F.interpolate(
                        images, size=(expected, expected), mode="bilinear", align_corners=False
                    )
            I_clip = vt(images_clip_in, output_hidden_states=False).last_hidden_state[:, 1:, :]
            I_dino_in = self.imagenet_norm(images)
            I_dino = self.dino_proj(self.dino(I_dino_in.float()).last_hidden_state)
            I_v, _ = self.cross_attention(I_clip, I_dino, I_dino)

        B = (images.size(0) if images is not None else I_v.size(0))
        learnable_q = self.learnable_query.unsqueeze(0).expand(B, -1, -1)
        q_input = torch.cat([learnable_q, text_embeds], dim=1) if text_embeds is not None else learnable_q
        q_input = self.q_proj_in(q_input)

        # 将密集图像 token 映射到 Q-Former 期望的 encoder_hidden_size
        I_v_kv = self.q_kv_proj(I_v) if self.q_kv_proj is not None else I_v

        # 确保 dtype/device 与 Q-Former 一致
        try:
            qf_param = next(self.qformer.parameters())
            q_device, q_dtype = qf_param.device, qf_param.dtype
        except StopIteration:
            q_device, q_dtype = self.device, q_input.dtype
        q_input = q_input.to(device=q_device, dtype=q_dtype)
        I_v_kv = I_v_kv.to(device=q_device, dtype=q_dtype)

        # 一次性调试信息与形状断言（帮助定位维度错配）
        if not self._qformer_debug_once:
            clip_dim = self._clip_dim
            q_dim = getattr(self.qformer.config, "hidden_size", None)
            kv_dim = getattr(self.qformer.config, "encoder_hidden_size", q_dim)
            print(f"[QFormer Debug] clip_dim={clip_dim}, qformer_q_dim={q_dim}, qformer_kv_dim={kv_dim}")
            print(f"[QFormer Debug] q_input={tuple(q_input.shape)}, I_v_kv={tuple(I_v_kv.shape)}")
            self._qformer_debug_once = True

        assert (
            q_input.shape[-1] == getattr(self.qformer.config, "hidden_size")
        ), f"Q-Former query dim mismatch: {q_input.shape[-1]} vs {self.qformer.config.hidden_size}"
        assert (
            I_v_kv.shape[-1] == getattr(self.qformer.config, "encoder_hidden_size", self.qformer.config.hidden_size)
        ), f"Q-Former KV dim mismatch: {I_v_kv.shape[-1]} vs {getattr(self.qformer.config, 'encoder_hidden_size', self.qformer.config.hidden_size)}"

        qf_out = self.qformer(query_embeds=q_input, encoder_hidden_states=I_v_kv).last_hidden_state
        img_embeds = qf_out[:, :self.learnable_query.size(0), :]
        target_dtype = (images.dtype if images is not None else q_dtype)
        img_embeds = self.q_proj_out(img_embeds).to(target_dtype)
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
