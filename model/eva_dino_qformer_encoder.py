import torch
import torch.nn as nn
from types import SimpleNamespace
from transformers import CLIPVisionModel, Blip2QFormerModel

# ============== 辅助：规范化与反规范化（近似） ==============
# 假设 images 已按 CLIP 规范化到 [-1,1]（很多 LLaVA/EVA-CLIP 管线如此）
# 将其近似还原到 [0,1]，再做 ImageNet 规范化供 DINO/ViT 使用
class OptionalReNorm(nn.Module):
    def __init__(self, enable=True):
        super().__init__()
        self.enable = enable
        # ImageNet 规范化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x_clip_norm):
        if not self.enable:
            return x_clip_norm
        # 近似把 [-1,1] 还原为 [0,1]
        x01 = (x_clip_norm + 1.0) * 0.5
        # 再做 ImageNet 规范化
        return (x01 - self.mean) / self.std

# ============== DINOv2 后端适配器 ==============
class DINOBackbone(nn.Module):
    """
    统一接口：
      forward(images) -> SimpleNamespace(last_hidden_state=[B, N, D])
    优先级：
      1) timm 的 DINOv2
      2) torch.hub facebookresearch/dinov2
      3) timm 的通用 ViT 兜底
      4) 恒等伪模型兜底（返回零特征）
    """
    def __init__(self, model_name='vit_large_patch14_dinov2.lvd142m',
                 use_timm=True, use_torchhub=True, imagenet_norm=True):
        super().__init__()
        self.imagenet_norm = OptionalReNorm(enable=imagenet_norm)
        self.backend = None
        self.embed_dim = None

        # 1) timm DINOv2
        if use_timm:
            try:
                import timm
                self.backend = timm.create_model(model_name, pretrained=True)
                # timm ViT 都有 embed_dim / num_patches 等属性
                self.embed_dim = getattr(self.backend, 'embed_dim', None)
                self._backend_type = 'timm_dino'
            except Exception:
                self.backend = None

        # 2) torch.hub 官方 dinov2
        if self.backend is None and use_torchhub:
            try:
                # 可选：'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
                self.backend = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
                # 常见 embed_dim：vitl14 为 1024
                self.embed_dim = getattr(self.backend, 'embed_dim', 1024)
                self._backend_type = 'hub_dino'
            except Exception:
                self.backend = None

        # 3) timm ViT 兜底
        if self.backend is None:
            try:
                import timm
                self.backend = timm.create_model('vit_large_patch14_224', pretrained=True)
                self.embed_dim = getattr(self.backend, 'embed_dim', None)
                self._backend_type = 'timm_vit'
            except Exception:
                self.backend = None

        # 4) 最终兜底：恒等伪模型（返回零向量）
        if self.backend is None or self.embed_dim is None:
            self._backend_type = 'dummy'
            self.embed_dim = 1024  # 给一个合理的默认 D
            # 注册一个“空参数”，以便模型能 .to(device)
            self.register_parameter('_dummy', nn.Parameter(torch.zeros(1)))

    @torch.inference_mode()
    def _forward_backend(self, x):
        t = self._backend_type
        if t in ('timm_dino', 'timm_vit'):
            # timm ViT 家族
            feats = self.backend.forward_features(x)  # 兼容新旧 timm：可能是 tensor 或 dict
            if isinstance(feats, dict):
                # 新版 timm 通常在 'x' 给出 [B, N+1, D]（含 CLS）
                tokens = feats.get('x', None)
                if tokens is None:
                    # 旧版可能直接返回 [B, D] 的 pooled，尝试从 'attn' 或其他键恢复困难
                    # 退化为使用全局 token 复制，避免崩溃
                    pooled = feats.get('pooled', None)
                    if pooled is not None:
                        tokens = pooled.unsqueeze(1)  # [B,1,D]
                    else:
                        raise RuntimeError('Unexpected timm forward_features dict structure')
            else:
                # 某些版本直接返回 [B, N+1, D] 的 tokens
                tokens = feats
            # 去掉 CLS
            if tokens.dim() == 3 and tokens.size(1) >= 2:
                tokens = tokens[:, 1:, :]
            return tokens  # [B, N, D]

        elif t == 'hub_dino':
            # 官方 dinov2：不同 commit 的字段名可能不同，这里尽量兼容
            if hasattr(self.backend, 'forward_features'):
                feats = self.backend.forward_features(x)
                if isinstance(feats, dict):
                    # 常见键：'x_norm_patchtokens'（[B,N,D]）或 'x_norm_clstoken'
                    tokens = feats.get('x_norm_patchtokens', None)
                    if tokens is None:
                        # 尝试 'x' 然后去 CLS
                        tokens = feats.get('x', None)
                        if tokens is not None and tokens.dim() == 3 and tokens.size(1) >= 2:
                            tokens = tokens[:, 1:, :]
                    if tokens is None:
                        raise RuntimeError('Unexpected hub dinov2 dict structure')
                else:
                    # 偶见直接返回 [B,N+1,D]
                    tokens = feats
                    if tokens.dim() == 3 and tokens.size(1) >= 2:
                        tokens = tokens[:, 1:, :]
                return tokens  # [B, N, D]
            else:
                # 极老版本：直接前向得到 [B, num_classes]，无法拿 tokens，只能报错
                raise RuntimeError('hub dinov2 has no forward_features')

        elif t == 'dummy':
            B, _, H, W = x.shape
            N = (H // 14) * (W // 14)  # 假定 patch14，做个形状兜底
            return x.new_zeros(B, N, self.embed_dim)

        else:
            raise RuntimeError(f'Unknown backend type: {t}')

    def forward(self, images):
        # 可选从 CLIP 规范化近似还原并转到 ImageNet 规范化
        x = self.imagenet_norm(images)  # 如果你已单独做了 ImageNet 规范化，可把 imagenet_norm=False
        tokens = self._forward_backend(x)  # [B, N, D]
        return SimpleNamespace(last_hidden_state=tokens, hidden_states=None)
        

class EvaDinoQFormerVisionTower(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 1) EVA-CLIP
        self.eva_clip = CLIPVisionModel.from_pretrained(args.eva_clip_path)
        clip_dim = self.eva_clip.config.hidden_size  # 例如 768/1024

        # 2) DINOv2 替代（优先 timm）
        #    model_name 可选：'vit_base_patch14_dinov2' 族或带权重后缀的 timm 名称
        dino_model_name = getattr(args, 'dino_model_name', 'vit_large_patch14_dinov2.lvd142m')
        self.dino = DINOBackbone(model_name=dino_model_name,
                                 use_timm=getattr(args, 'use_timm', True),
                                 use_torchhub=getattr(args, 'use_torchhub', True),
                                 imagenet_norm=getattr(args, 'dino_use_imagenet_norm', True))
        dino_dim = self.dino.embed_dim

        # 维度对齐：将 DINO 的 D 投到 EVA-CLIP 的 D
        self.dino_proj = nn.Linear(dino_dim, clip_dim, bias=False)

        # 3) Cross Attention (EVA 作为 Q，DINO 作为 K/V)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=clip_dim, num_heads=8, batch_first=True
        )

        # 4) Q-Former
        self.qformer = Blip2QFormerModel.from_pretrained(args.qformer_path)
        assert self.qformer.config.hidden_size == clip_dim, \
            f"Q-Former hidden_size({self.qformer.config.hidden_size})需与 clip_dim({clip_dim})一致"

        # 5) Learnable Queries（用于 Q-Former 的 query_embeds）
        self.learnable_query = nn.Parameter(torch.randn(args.num_query, clip_dim))

        # 可选：LayerNorm 稳定跨模态融合
        self.pre_ln_clip = nn.LayerNorm(clip_dim)
        self.pre_ln_dino = nn.LayerNorm(clip_dim)

    def forward(self, images, text_embeds=None):
        # images：建议已按 CLIP 规范化到 [-1,1]；DINO 支路内部会近似转换到 ImageNet 规范化
        # 1) 视觉特征
        clip_out = self.eva_clip(images, output_hidden_states=True)
        I_clip = clip_out.last_hidden_state            # [B, N1, D]
        dino_out = self.dino(images)
        I_dino_raw = dino_out.last_hidden_state        # [B, N2, D_dino]
        I_dino = self.dino_proj(I_dino_raw)            # [B, N2, D_clip]

        # 2) 可选 LayerNorm
        I_clip = self.pre_ln_clip(I_clip)
        I_dino = self.pre_ln_dino(I_dino)

        # 3) Cross Attention：EVA 为 Query，DINO 为 K/V
        I_v, _ = self.cross_attention(I_clip, I_dino, I_dino)  # [B, N1, D]

        # 4) 组织 Q-Former 的 query_embeds
        B = images.size(0)
        learnable_q = self.learnable_query.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        if text_embeds is not None:
            q_input = torch.cat([learnable_q, text_embeds], dim=1)         # [B, Q+Tq, D]
        else:
            q_input = learnable_q

        # 5) Q-Former 融合
        qf_out = self.qformer(
            query_embeds=q_input,
            encoder_hidden_states=I_v
        ).last_hidden_state  # [B, Q(+Tq), D]

        # 6) 仅取 learnable_query 对应部分作为最终视觉特征（与原逻辑一致）
        img_embeds = qf_out[:, :self.learnable_query.size(0), :]  # [B, Q, D]
        return img_embeds
