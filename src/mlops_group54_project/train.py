from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from mlops_group54_project.model import ModelConfig, build_model


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_3ch(x: torch.Tensor) -> torch.Tensor:
    # ResNet50 expects 3 channels
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")


def brain_tumor_dataset(processed_dir: Path) -> tuple[TensorDataset, TensorDataset]:
    train_images = torch.load(processed_dir / "train_images.pt")
    train_labels = torch.load(processed_dir / "train_labels.pt")
    test_images = torch.load(processed_dir / "test_images.pt")
    test_labels = torch.load(processed_dir / "test_labels.pt")

    train_images = _ensure_3ch(train_images)
    test_images = _ensure_3ch(test_images)

    train_set = TensorDataset(train_images, train_labels)
    test_set = TensorDataset(test_images, test_labels)
    return train_set, test_set


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    device = _device()

    # Load data
    processed_dir = Path(cfg.data.processed_dir)
    train_set, _ = brain_tumor_dataset(processed_dir)

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(getattr(cfg.data, "num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    # Build model (ResNet50)
    model_cfg = ModelConfig(
        backbone=str(cfg.model.backbone),
        pretrained=bool(cfg.model.pretrained),
        num_classes=int(cfg.model.num_classes),
        dropout=float(getattr(cfg.model, "dropout", 0.0)),
    )
    model = build_model(model_cfg).to(device)

    # Train
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(getattr(cfg.train, "weight_decay", 0.0)),
    )

    epochs = int(cfg.train.epochs)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            running_acc += (logits.argmax(dim=1) == y).float().sum().item()
            n += bs

        epoch_loss = running_loss / max(n, 1)
        epoch_acc = running_acc / max(n, 1)

        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

    # Save
    out_dir = Path(getattr(cfg.train, "output_dir", "models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / str(getattr(cfg.train, "checkpoint_name", "model.pth")))


if __name__ == "__main__":
    train()
