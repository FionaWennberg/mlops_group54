from pathlib import Path
import pytest
from torch.utils.data import Dataset

from mlops_group54_project.data import FolderDataset, build_preprocess_transform


def test_folder_dataset_is_dataset_when_path_exists(tmp_path: Path):
    # Create a minimal ImageFolder structure:
    # tmp_path/class_a/img.jpg
    class_dir = tmp_path / "class_a"
    class_dir.mkdir(parents=True)

    # Create a dummy "image" file (ImageFolder expects an image extension).
    # It will fail to load if we actually index it, but we can still test construction/typing.
    (class_dir / "dummy.jpg").write_bytes(b"not a real jpg")

    tf = build_preprocess_transform(image_size=224)
    ds = FolderDataset(root=tmp_path, transform=tf)

    assert isinstance(ds, Dataset)


def test_folder_dataset_raises_when_missing_path():
    tf = build_preprocess_transform(image_size=224)
    with pytest.raises(FileNotFoundError):
        FolderDataset(root=Path("data/this_does_not_exist"), transform=tf)
