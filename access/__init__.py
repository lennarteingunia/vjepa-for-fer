import logging
from typing import Any, Dict, Optional, Union
import torch
import yaml

from models import vision_transformer
from models.attentive_pooler import AttentiveClassifier


def build_encoder(
    config: str,
    *,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_key: str = 'target_encoder',
) -> vision_transformer.VisionTransformer:
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_name = config.get('pretrain').get('model_name', None)
    model = vision_transformer.__dict__[model_name](
        img_size=config.get('optimization').get('resolution', 224),
        patch_size=config.get('pretrain').get('patch_size', None),
        num_frames=config.get('pretrain').get('frames_per_clip', 1),
        tubelet_size=config.get('pretrain').get('tubelet_size', 2),
        uniform_power=config.get('pretrain').get('uniform_power', False),
        use_sdpa=config.get('pretrain').get('use_sdpa', True),
        use_SiLU=config.get('pretrain').get('use_silu', False),
        tight_SiLU=config.get('pretrain').get('tight_silu', True),
    )
    if checkpoint_path is not None:
        encoder = load_encoder_weights(model, checkpoint_path)
    if device is not None:
        encoder = encoder.to(device)
    return encoder


def load_encoder_weights(
    encoder: torch.nn.Module,
    checkpoint_path: str,
    *,
    checkpoint_key: str = 'encoder',
    logger: logging.Logger = logging.getLogger(__name__),
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    checkpoint = checkpoint[checkpoint_key]
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    checkpoint = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}
    msg = encoder.load_state_dict(checkpoint)
    logger.info(f'Loaded encoder with message: {msg}')
    del checkpoint
    return encoder


def build_attentive_classifier(
    config: str,
    *,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> AttentiveClassifier:
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    embed_dim = vision_transformer.VIT_EMBED_DIMS[config.get(
        'pretrain').get('model_name', None)]
    num_heads = VIT_NUM_HEADS[config.get('pretrain').get('model_name', None)]
    num_classes = config.get('data').get('num_classes', None)
    model = AttentiveClassifier(
        embed_dim=embed_dim, num_heads=num_heads, depth=1, num_classes=num_classes)
    if checkpoint_path is not None:
        model = load_attentive_classifier_weights(
            model, checkpoint_path=checkpoint_path)
    if device is not None:
        model = model.to(device)
    return model


def load_attentive_classifier_weights(
    attentive_classsifier: AttentiveClassifier,
    checkpoint_path: str,
    *,
    logger: logging.Logger = logging.getLogger(__name__),
    remove_module_from_entry_key: bool = True
) -> AttentiveClassifier:
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu')
    classifier_checkpoint = checkpoint['classifier']
    if remove_module_from_entry_key:
        checkpoint_path = {
            k.replace('module.', ''): v for k, v in classifier_checkpoint.items()}
    load_msg = attentive_classsifier.load_state_dict(
        checkpoint_path, strict=False)
    logger.info(f'Loaded attentive classifier with msg: {load_msg}')
    del checkpoint
    return attentive_classsifier


# This is taken from the construction classes in vision_transformer.py. We don't want to have to construct the whole vision transformer just to be able to construct the attentive classifier.
VIT_NUM_HEADS = {
    'vit_tiny': 3,
    'vit_small': 6,
    'vit_base': 12,
    'vit_large': 16,
    'vit_huge': 16,
    'vit_giant': 16,
    'vit_gigantic': 16,
}
