from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import wandb
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


def brain_tumor_dataset(processed_dir: Path) -> Tuple[TensorDataset, TensorDataset]:
    train_images = torch.load(processed_dir / "train_images.pt")
    train_labels = torch.load(processed_dir / "train_labels.pt")
    test_images = torch.load(processed_dir / "test_images.pt")
    test_labels = torch.load(processed_dir / "test_labels.pt")

    train_images = _ensure_3ch(train_images)
    test_images = _ensure_3ch(test_images)

    train_set = TensorDataset(train_images, train_labels)
    test_set = TensorDataset(test_images, test_labels)
    return train_set, test_set


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    """
    Resolve checkpoint path from config.

    Supported patterns:
      - cfg.eval.checkpoint_path (absolute or relative)
      - fallback to cfg.train.output_dir / cfg.train.checkpoint_name
      - fallback to models/model.pth
    """
    # Prefer explicit eval.checkpoint_path
    if "eval" in cfg and getattr(cfg.eval, "checkpoint_path", None):
        return Path(cfg.eval.checkpoint_path)

    # Fallback to train output location
    out_dir = Path(getattr(cfg.train, "output_dir", "models"))
    ckpt_name = str(getattr(cfg.train, "checkpoint_name", "model.pth"))
    return out_dir / ckpt_name


@dataclass(frozen=True)
class EvalResults:
    loss: float
    accuracy: float
    num_examples: int
    confusion_matrix: torch.Tensor  # shape [C, C]
    per_class_accuracy: Dict[int, float]


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> EvalResults:
    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    n = 0

    # Confusion matrix: rows = true, cols = pred
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    model.eval()
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        preds = logits.argmax(dim=1)

        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (preds == y).sum().item()
        n += bs

        # Update confusion matrix on CPU for simplicity
        for t, p in zip(y.view(-1).tolist(), preds.view(-1).tolist()):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1

    avg_loss = total_loss / max(n, 1)
    acc = correct / max(n, 1)

    # Per-class accuracy: diag / row_sum
    per_class_acc: Dict[int, float] = {}
    for c in range(num_classes):
        row_sum = cm[c, :].sum().item()
        per_class_acc[c] = (cm[c, c].item() / row_sum) if row_sum > 0 else 0.0

    return EvalResults(
        loss=avg_loss,
        accuracy=acc,
        num_examples=n,
        confusion_matrix=cm,
        per_class_accuracy=per_class_acc,
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def evaluate(cfg: DictConfig) -> Dict[str, Any]:
    device = _device()

    # Load data (test set)
    processed_dir = Path(cfg.data.processed_dir)
    _, test_set = brain_tumor_dataset(processed_dir)

    test_loader = DataLoader(
        test_set,
        batch_size=int(getattr(cfg.eval, "batch_size", getattr(cfg.train, "batch_size", 32)))
        if "eval" in cfg
        else int(getattr(cfg.train, "batch_size", 32)),
        shuffle=False,
        num_workers=int(getattr(cfg.data, "num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    # Build model
    model_cfg = ModelConfig(
        backbone=str(cfg.model.backbone),
        pretrained=bool(cfg.model.pretrained),
        num_classes=int(cfg.model.num_classes),
        dropout=float(getattr(cfg.model, "dropout", 0.0)),
    )
    model = build_model(model_cfg).to(device)

    # Load checkpoint
    ckpt_path = _resolve_checkpoint_path(cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path.resolve()}\n"
            "Set cfg.eval.checkpoint_path or ensure cfg.train.output_dir/cfg.train.checkpoint_name exists."
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    # Evaluate
    results = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=int(cfg.model.num_classes),
    )

    # Prepare metrics dict (W&B-friendly later)
    metrics: Dict[str, Any] = {
        "eval/loss": results.loss,
        "eval/accuracy": results.accuracy,
        "eval/num_examples": results.num_examples,
    }

    # Add per-class accuracy as separate scalars
    for cls_idx, cls_acc in results.per_class_accuracy.items():
        metrics[f"eval/per_class_accuracy_{cls_idx}"] = cls_acc

    # --- W&B logging (optional) ---
    # --- W&B logging (optional) ---
    if "eval" in cfg and hasattr(cfg.eval, "wandb") and cfg.eval.wandb.get("enabled", False):
        wandb.init(
            project=str(cfg.eval.wandb.project),
            entity=(None if cfg.eval.wandb.entity in [None, "null"] else str(cfg.eval.wandb.entity)),
            name=(None if cfg.eval.wandb.run_name in [None, "null"] else str(cfg.eval.wandb.run_name)),
            tags=list(getattr(cfg.eval.wandb, "tags", [])),
            config={
                "model": {
                    "backbone": str(cfg.model.backbone),
                    "pretrained": bool(cfg.model.pretrained),
                    "num_classes": int(cfg.model.num_classes),
                    "dropout": float(getattr(cfg.model, "dropout", 0.0)),
                },
                "data": {
                    "processed_dir": str(cfg.data.processed_dir),
                },
                "eval": {
                    "batch_size": int(getattr(cfg.eval, "batch_size", 32)),
                    "checkpoint_path": str(ckpt_path),
                },
            },
        )


        # log scalar metrics
        wandb.log(metrics)

        # log confusion matrix (as a table)
        cm = results.confusion_matrix.cpu().numpy()

        table = wandb.Table(
            columns=["true_class"] + [f"pred_{i}" for i in range(cm.shape[1])],
        )

        for true_idx in range(cm.shape[0]):
            table.add_data(true_idx, *cm[true_idx].tolist())
        wandb.log({"eval/confusion_matrix": table})

        wandb.finish()

    # Print a compact summary (CI/log friendly)
    print(f"Eval - loss: {results.loss:.4f} - acc: {results.accuracy:.4f} - n: {results.num_examples}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(results.confusion_matrix)

    return metrics


if __name__ == "__main__":
    evaluate()
