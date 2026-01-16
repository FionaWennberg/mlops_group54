from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

import mlops_group54_project.data as data_mod


def test_build_preprocess_transform_returns_compose():
    tf = data_mod.build_preprocess_transform(image_size=224)
    assert hasattr(tf, "transforms")
    names = [t.__class__.__name__ for t in tf.transforms]
    assert names == ["Resize", "ToTensor", "Normalize"]


def test_folderdataset_raises_if_path_missing(tmp_path):
    missing = tmp_path / "does_not_exist"
    tf = data_mod.build_preprocess_transform(image_size=224)
    with pytest.raises(FileNotFoundError, match="Dataset path does not exist"):
        data_mod.FolderDataset(root=missing, transform=tf)


def test_folderdataset_constructs_with_minimal_imagefolder_structure(tmp_path: Path):
    """
    Light integration:
    - Opretter en minimal ImageFolder struktur med et rigtigt billede.
    - Tester at FolderDataset kan konstrueres og har classes/mapping.
    """
    try:
        from PIL import Image
    except Exception:
        pytest.skip("PIL/Pillow not available in this environment")

    class_dir = tmp_path / "class_a"
    class_dir.mkdir(parents=True)

    # Create a tiny valid JPEG image
    img_path = class_dir / "img1.jpg"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path)

    tf = data_mod.build_preprocess_transform(image_size=224)
    ds = data_mod.FolderDataset(root=tmp_path, transform=tf)

    assert isinstance(ds, Dataset)
    assert len(ds) == 1
    assert ds.classes == ["class_a"]
    assert ds.class_to_idx == {"class_a": 0}


def test_stack_dataset_to_tensors_shapes():
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, n=5):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            x = torch.randn(3, 8, 8)
            y = torch.tensor(idx, dtype=torch.long)
            return x, y

    ds = DummyDataset(n=5)
    images, labels = data_mod._stack_dataset_to_tensors(ds, batch_size=2, num_workers=0)

    assert images.shape == (5, 3, 8, 8)
    assert labels.shape == (5,)
    assert torch.equal(labels, torch.arange(0, 5))


def test_preprocess_and_save_happy_path_mocks(tmp_path, monkeypatch):
    # capture torch.save calls without writing files
    save_calls = []

    def fake_torch_save(obj, path):
        save_calls.append(Path(path).name)

    monkeypatch.setattr(data_mod.torch, "save", fake_torch_save)

    # fake datasets returned by FolderDataset
    class FakeDS:
        def __init__(self, classes, mapping):
            self._classes = classes
            self._mapping = mapping

        @property
        def classes(self):
            return self._classes

        @property
        def class_to_idx(self):
            return self._mapping

    mapping = {"a": 0, "b": 1}
    train_ds = FakeDS(classes=["a", "b"], mapping=mapping)
    test_ds = FakeDS(classes=["a", "b"], mapping=mapping)
    created = {"count": 0}

    def fake_folderdataset(root, transform):
        created["count"] += 1
        return train_ds if created["count"] == 1 else test_ds

    monkeypatch.setattr(data_mod, "FolderDataset", fake_folderdataset)

    # mock stacking to avoid DataLoader/ImageFolder
    def fake_stack(ds, batch_size, num_workers):
        imgs = torch.zeros(2, 3, 4, 4)
        lbls = torch.tensor([0, 1], dtype=torch.long)
        return imgs, lbls

    monkeypatch.setattr(data_mod, "_stack_dataset_to_tensors", fake_stack)

    cfg = data_mod.DataConfig(
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        processed_dir=tmp_path / "processed",
        image_size=224,
        preprocess_batch_size=64,
        num_workers=0,
    )

    data_mod.preprocess_and_save(cfg)

    assert sorted(save_calls) == sorted(
        [
            "train_images.pt",
            "train_labels.pt",
            "test_images.pt",
            "test_labels.pt",
            "class_mapping.pt",
        ]
    )


def test_preprocess_and_save_raises_on_mapping_mismatch(tmp_path, monkeypatch):
    class FakeDS:
        def __init__(self, classes, mapping):
            self._classes = classes
            self._mapping = mapping

        @property
        def classes(self):
            return self._classes

        @property
        def class_to_idx(self):
            return self._mapping

    train_ds = FakeDS(classes=["a", "b"], mapping={"a": 0, "b": 1})
    test_ds = FakeDS(classes=["a", "c"], mapping={"a": 0, "c": 1})

    created = {"count": 0}

    def fake_folderdataset(root, transform):
        created["count"] += 1
        return train_ds if created["count"] == 1 else test_ds

    monkeypatch.setattr(data_mod, "FolderDataset", fake_folderdataset)

    cfg = data_mod.DataConfig(
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        processed_dir=tmp_path / "processed",
        image_size=224,
        preprocess_batch_size=64,
        num_workers=0,
    )

    with pytest.raises(RuntimeError, match="Train and test class mappings differ"):
        data_mod.preprocess_and_save(cfg)


def test_to_data_config_reads_fields_and_defaults():
    cfg = OmegaConf.create(
        {
            "data": {
                "train_dir": "data/raw/Training",
                "test_dir": "data/raw/Testing",
                "processed_dir": "data/processed",
                "image_size": 224,
                "num_workers": 4,
                "preprocess_batch_size": 64,
            }
        }
    )
    dc = data_mod._to_data_config(cfg)
    assert dc.train_dir == Path("data/raw/Training")
    assert dc.test_dir == Path("data/raw/Testing")
    assert dc.processed_dir == Path("data/processed")
    assert dc.image_size == 224
    assert dc.num_workers == 4
    assert dc.preprocess_batch_size == 64

    # without preprocess_batch_size -> default 64
    cfg2 = OmegaConf.create(
        {
            "data": {
                "train_dir": "x",
                "test_dir": "y",
                "processed_dir": "z",
                "image_size": 128,
                "num_workers": 0,
            }
        }
    )
    dc2 = data_mod._to_data_config(cfg2)
    assert dc2.preprocess_batch_size == 64
