# Data stage: læser rå billeder fra class-folders og gemmer deterministiske tensors i data/processed.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


# ----------------
# Config dataclass
# ----------------
@dataclass(frozen=True)
class DataConfig:
    train_dir: Path
    test_dir: Path
    processed_dir: Path
    image_size: int = 224
    preprocess_batch_size: int = 64
    num_workers: int = 4


# -----------------------
# Deterministic transform
# -----------------------
def build_preprocess_transform(image_size: int) -> transforms.Compose:
    # Deterministisk preprocessing (ingen random augmentation) for reproducerbare .pt-filer.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# -----------------------
# Dataset + tensor export
# -----------------------
class FolderDataset(Dataset):
    """Wrapper around ImageFolder so we can expose class mappings cleanly."""

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        if not root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {root.resolve()}")
        self.ds = ImageFolder(root=str(root), transform=transform)

    @property
    def classes(self) -> List[str]:
        return self.ds.classes

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return self.ds.class_to_idx

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int):
        return self.ds[index]


def _stack_dataset_to_tensors(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Loader hele datasættet og stakker batches til (images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []

    for x, y in loader:
        xs.append(x)
        ys.append(y)

    images = torch.cat(xs, dim=0)
    labels = torch.cat(ys, dim=0)
    return images, labels


def preprocess_and_save(cfg: DataConfig) -> None:
    """
    Reads raw folder data, applies deterministic preprocessing, and saves:
      processed_dir/train_images.pt
      processed_dir/train_labels.pt
      processed_dir/test_images.pt
      processed_dir/test_labels.pt
      processed_dir/class_mapping.pt
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    tf = build_preprocess_transform(cfg.image_size)

    train_ds = FolderDataset(cfg.train_dir, transform=tf)
    test_ds = FolderDataset(cfg.test_dir, transform=tf)

    if train_ds.class_to_idx != test_ds.class_to_idx:
        raise RuntimeError(
            "Train and test class mappings differ.\n"
            f"Train mapping: {train_ds.class_to_idx}\n"
            f"Test mapping:  {test_ds.class_to_idx}\n\n"
            "Make sure both raw folders contain the same class subfolder names."
        )

    train_images, train_labels = _stack_dataset_to_tensors(
        train_ds,
        batch_size=cfg.preprocess_batch_size,
        num_workers=cfg.num_workers,
    )
    test_images, test_labels = _stack_dataset_to_tensors(
        test_ds,
        batch_size=cfg.preprocess_batch_size,
        num_workers=cfg.num_workers,
    )

    # Gemmer tensors + class mapping til senere træning/evaluering
    torch.save(train_images, cfg.processed_dir / "train_images.pt")
    torch.save(train_labels, cfg.processed_dir / "train_labels.pt")
    torch.save(test_images, cfg.processed_dir / "test_images.pt")
    torch.save(test_labels, cfg.processed_dir / "test_labels.pt")

    torch.save(
        {"classes": train_ds.classes, "class_to_idx": train_ds.class_to_idx},
        cfg.processed_dir / "class_mapping.pt",
    )


# ----------------
# Hydra entrypoint
# ----------------
def _to_data_config(cfg: DictConfig) -> DataConfig:
    d = cfg.data
    return DataConfig(
        train_dir=Path(d.train_dir),
        test_dir=Path(d.test_dir),
        processed_dir=Path(d.processed_dir),
        image_size=int(d.image_size),
        preprocess_batch_size=int(getattr(d, "preprocess_batch_size", 64)),
        num_workers=int(d.num_workers),
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data_cfg = _to_data_config(cfg)
    preprocess_and_save(data_cfg)


if __name__ == "__main__":
    main()
