from pathlib import Path
import torch
import typer
from torchvision import datasets, transforms


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """
    Preprocess brain tumor image data and save tensors.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),   # standard size
            transforms.ToTensor(),           # [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=raw_dir / "Training",
        transform=transform,
    )

    test_dataset = datasets.ImageFolder(
        root=raw_dir / "Testing",
        transform=transform,
    )

    def dataset_to_tensors(dataset):
        images, labels = [], []
        for img, label in dataset:
            images.append(img)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)

    train_images, train_targets = dataset_to_tensors(train_dataset)
    test_images, test_targets = dataset_to_tensors(test_dataset)

    torch.save(train_images, processed_dir / "train_images.pt")
    torch.save(train_targets, processed_dir / "train_targets.pt")
    torch.save(test_images, processed_dir / "test_images.pt")
    torch.save(test_targets, processed_dir / "test_targets.pt")

    # Save class mapping (VERY important)
    torch.save(train_dataset.class_to_idx, processed_dir / "class_to_idx.pt")

    print("Preprocessing complete")
    print("Class mapping:", train_dataset.class_to_idx)

if __name__ == "__main__":
    typer.run(preprocess_data)
