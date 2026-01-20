from __future__ import annotations

# Train a brain tumor classifier (Hydra-configured).

from pathlib import Path
from typing import Any, Optional

import hydra
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from mlops_group54_project.model import ModelConfig, build_model


def _device() -> torch.device:
    # Prefer GPU if available (CUDA > MPS > CPU).
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_3ch(x: torch.Tensor) -> torch.Tensor:
    # ResNet backbones expect (N, 3, H, W).
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        # Repeat grayscale channel -> RGB-like.
        return x.repeat(1, 3, 1, 1)
    raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")


def brain_tumor_dataset(processed_dir: Path) -> tuple[TensorDataset, TensorDataset]:
    # Load preprocessed tensors from disk.
    train_images = torch.load(processed_dir / "train_images.pt")
    train_labels = torch.load(processed_dir / "train_labels.pt")
    test_images = torch.load(processed_dir / "test_images.pt")
    test_labels = torch.load(processed_dir / "test_labels.pt")

    train_images = _ensure_3ch(train_images)
    test_images = _ensure_3ch(test_images)

    train_set = TensorDataset(train_images, train_labels)
    test_set = TensorDataset(test_images, test_labels)
    return train_set, test_set


def _wandb_enabled(cfg: DictConfig) -> bool:
    return bool("train" in cfg and hasattr(cfg.train, "wandb") and cfg.train.wandb.get("enabled", False))


def _as_optional_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if str(x) == "null":
        return None
    s = str(x).strip()
    return None if s == "" else s


def _init_wandb(cfg: DictConfig) -> Optional[wandb.sdk.wandb_run.Run]:
    if not _wandb_enabled(cfg):
        return None

    # Default to logging every 20 steps if not set (we still read it again in the training loop).
    _ = int(getattr(cfg.train.wandb, "log_every_n_steps", 20))

    run = wandb.init(
        project=str(cfg.train.wandb.project),
        entity=_as_optional_str(getattr(cfg.train.wandb, "entity", None)),
        name=_as_optional_str(getattr(cfg.train.wandb, "run_name", None)),
        tags=list(getattr(cfg.train.wandb, "tags", [])),
        config={
            "model": {
                "backbone": str(cfg.model.backbone),
                "pretrained": bool(cfg.model.pretrained),
                "num_classes": int(cfg.model.num_classes),
                "dropout": float(getattr(cfg.model, "dropout", 0.0)),
            },
            "data": {
                "processed_dir": str(cfg.data.processed_dir),
                "batch_size": int(cfg.train.batch_size),
                "num_workers": int(getattr(cfg.data, "num_workers", 0)),
            },
            "train": {
                "lr": float(cfg.train.lr),
                "weight_decay": float(getattr(cfg.train, "weight_decay", 0.0)),
                "epochs": int(cfg.train.epochs),
                "output_dir": str(getattr(cfg.train, "output_dir", "models")),
                "checkpoint_name": str(getattr(cfg.train, "checkpoint_name", "model.pth")),
                "log_every_n_steps": int(getattr(cfg.train.wandb, "log_every_n_steps", 20)),
            },
        },
    )
    return run


@torch.no_grad()
def _evaluate_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on validation data (here: the provided test split) for one epoch.
    Returns (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((logits.argmax(dim=1) == y).sum().item())
        n += bs

    avg_loss = total_loss / max(n, 1)
    acc = correct / max(n, 1)
    return avg_loss, acc


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    device = _device()

    # --- W&B init (optional) ---
    run = _init_wandb(cfg)
    log_every = int(getattr(cfg.train.wandb, "log_every_n_steps", 20)) if run else 0

    # Data
    processed_dir = Path(cfg.data.processed_dir)
    train_set, test_set = brain_tumor_dataset(processed_dir)

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(getattr(cfg.data, "num_workers", 0)),
        # Faster host->GPU transfer on CUDA.
        pin_memory=(device.type == "cuda"),
    )

    # Validation loader uses the provided test split (no separate validation set)
    val_loader = DataLoader(
        test_set,
        batch_size=int(getattr(cfg.eval, "batch_size", cfg.train.batch_size)),
        shuffle=False,
        num_workers=int(getattr(cfg.data, "num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model_cfg = ModelConfig(
        backbone=str(cfg.model.backbone),
        pretrained=bool(cfg.model.pretrained),
        num_classes=int(cfg.model.num_classes),
        dropout=float(getattr(cfg.model, "dropout", 0.0)),
    )
    model = build_model(model_cfg).to(device)

    # Loss + optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(getattr(cfg.train, "weight_decay", 0.0)),
    )

    global_step = 0
    epochs = int(cfg.train.epochs)
    for epoch in range(epochs):
        # Enable training mode.
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward + backward + update.
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            correct = (logits.argmax(dim=1) == y).float().sum().item()

            # Track epoch metrics.
            running_loss += loss.item() * bs
            running_acc += correct
            n += bs

            global_step += 1

            # Console logging
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")

            # W&B step logging (optional)
            if run and (global_step % max(log_every, 1) == 0):
                wandb.log(
                    {
                        "train/loss_step": float(loss.item()),
                        "train/acc_step": float(correct / max(bs, 1)),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "epoch": int(epoch),
                    },
                    step=global_step,
                )

        train_epoch_loss = running_loss / max(n, 1)
        train_epoch_acc = running_acc / max(n, 1)

        # Validation on the provided test split (used here as validation)
        val_loss, val_acc = _evaluate_epoch(model=model, data_loader=val_loader, loss_fn=loss_fn, device=device)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"train_loss: {train_epoch_loss:.4f} - train_acc: {train_epoch_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

        # W&B epoch logging (optional)
        if run:
            wandb.log(
                {
                    "train/loss_epoch": float(train_epoch_loss),
                    "train/acc_epoch": float(train_epoch_acc),
                    "val/loss_epoch": float(val_loss),
                    "val/acc_epoch": float(val_acc),
                    "epoch": int(epoch),
                },
                step=global_step,
            )

    # Save weights
    out_dir = Path(getattr(cfg.train, "output_dir", "models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / str(getattr(cfg.train, "checkpoint_name", "model.pth"))
    torch.save(model.state_dict(), ckpt_path)

    # Log checkpoint as a W&B artifact (optional, but useful)
    if run:
        artifact = wandb.Artifact(name="model", type="checkpoint")
        artifact.add_file(str(ckpt_path))
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    train()
