from __future__ import annotations

from dataclasses import dataclass

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 4
    dropout: float = 0.0  # optional dropout before final fc


def build_model(cfg: ModelConfig) -> nn.Module:
    """
    Build a classification model (logits output).
    For ResNet50, replaces the final fully-connected layer to match num_classes.
    """
    if cfg.backbone != "resnet50":
        raise ValueError(f"Unsupported backbone: {cfg.backbone}. Expected 'resnet50'.")

    # Torchvision API: newer versions use weights=...
    # We handle both pretrained styles safely.
    try:
        weights = models.ResNet50_Weights.DEFAULT if cfg.pretrained else None
        model = models.resnet50(weights=weights)
    except TypeError:
        # Fallback for older torchvision
        model = models.resnet50(pretrained=cfg.pretrained)

    in_features = model.fc.in_features

    if cfg.dropout and cfg.dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(in_features, cfg.num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, cfg.num_classes)

    return model


def _to_model_config(cfg: DictConfig) -> ModelConfig:
    m = cfg.model
    return ModelConfig(
        backbone=str(m.backbone),
        pretrained=bool(m.pretrained),
        num_classes=int(m.num_classes),
        dropout=float(getattr(m, "dropout", 0.0)),
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Simple smoke test: build model and move to device
    model_cfg = _to_model_config(cfg)
    _ = build_model(model_cfg).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
