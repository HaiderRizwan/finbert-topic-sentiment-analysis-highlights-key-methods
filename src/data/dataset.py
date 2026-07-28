from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import cfg


def get_dataloaders(
    dataset_root: Path = Path(cfg.PROJECT_ROOT) / "dataset_split",
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    Applies appropriate data augmentation for training, and
    standard resizing/normalization for validation/testing.
    """
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    if not all(d.exists() for d in [train_dir, val_dir, test_dir]):
        raise FileNotFoundError(
            f"Dataset splits not found in {dataset_root}. Ensure they are generated."
        )

    # ImageNet normalization standard (used by MobileNet and NASNet)
    normalize = transforms.Normalize(mean=cfg.frames.mean, std=cfg.frames.std)
    target_size = cfg.frames.target_size

    # Stronger data augmentation for the train set to improve robustness
    train_transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Validation and Test sets only get resized and normalized
    val_test_transform = transforms.Compose(
        [transforms.Resize(target_size), transforms.ToTensor(), normalize]
    )

    # Instantiate datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(
        root=str(test_dir), transform=val_test_transform
    )

    # Class mappings - ensuring consistency
    class_names = train_dataset.classes
    print(f"Discovered classes: {class_names}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names
