from .clip_encoder import CLIPVisionTower
from model.eva_dino_qformer_encoder import EvaDinoQFormerVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    # 强制使用自定义的 EVA+DINO+Q-Former 视觉塔，以满足新融合策略
    # 保留回退开关：若环境需原生 CLIP，可按需改回原逻辑
    return EvaDinoQFormerVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
