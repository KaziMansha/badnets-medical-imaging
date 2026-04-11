"""
Preprocessing for the Brain Tumor MRI Dataset (Nickparvar, 4-class).
"""

import csv
import json
import os
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

#Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "Training"
RAW_TEST_DIR = PROJECT_ROOT / "data" / "raw" / "Testing"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

#Constants
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VAL_FRACTION = 0.20
RANDOM_SEED = 42 #fixed so splits are reproducible across group members

#Sorted alphabetically so class indices are consistent
CLASS_NAMES = sorted(["glioma", "meningioma", "notumor", "pituitary"])
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

#Transforms
def get_transform(split):
    """
    Transforms for each split.

    Training split gets a random flip for augmentation
    """
    base = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if split == "train":
        base.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(base)

#Dataset
class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI dataset

    Each sample is a pair of (image_tensor, label)
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

#Split Helpers
def _collect_samples(root_dir):
    """
    Walk root/<class_name>/<image_file> and return a list of tuples of (project-relative POSIX path str, label_index)

    Paths are stored relative to PROJECT_ROOT so CSVs work across group members
    """
    samples = []
    for class_name, label in CLASS_TO_IDX.items():
        class_dir = root_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory not found: {class_dir}")
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                rel_path = img_file.relative_to(PROJECT_ROOT).as_posix()
                samples.append((rel_path, label))
    return samples


def save_splits(val_fraction = VAL_FRACTION, seed = RANDOM_SEED):
    """
    Build train/val/test splits and save them to data/splits/.

    Writes:
    train.csv, val.csv, test.csv, class_map.json
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    all_train = _collect_samples(RAW_TRAIN_DIR)
    paths = [s[0] for s in all_train]
    labels = [s[1] for s in all_train]

    #Stratify will ensure that each split has a roughly equal class distribution
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )

    test_samples = _collect_samples(RAW_TEST_DIR)
    test_paths = [s[0] for s in test_samples]
    test_labels = [s[1] for s in test_samples]

    def _write_csv(filepath, paths, labels):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            for p, l in zip(paths, labels):
                writer.writerow([p, l])

    _write_csv(SPLITS_DIR / "train.csv", train_paths, train_labels)
    _write_csv(SPLITS_DIR / "val.csv", val_paths, val_labels)
    _write_csv(SPLITS_DIR / "test.csv", test_paths, test_labels)

    #Save class map for reference during training and evaluation
    class_map = {
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {str(v): k for k, v in CLASS_TO_IDX.items()},
    }
    with open(SPLITS_DIR / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    print(f"Splits saved — train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)}")


def load_split(split):
    """
    Read a saved split CSV and return a list of tuples of (absolute path, label)
    """
    csv_path = SPLITS_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split file not found: {csv_path}. Run save_splits() first.")
    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abs_path = str(PROJECT_ROOT / row["image_path"])
            samples.append((abs_path, int(row["label"])))
    return samples

#DataLoader Factory
def get_dataloaders(batch_size = 32, num_workers = 0, pin_memory = False):
    """
    Build DataLoaders for all three splits
    
    Expects split CSVs to already exist (If not the case, call save_splits() first)
    """
    loaders = {}
    for split in ("train", "val", "test"):
        samples = load_split(split)
        dataset = BrainTumorDataset(samples, transform=get_transform(split))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders

#Entry Point
if __name__ == "__main__":
    print("Building and saving dataset splits...")
    save_splits()

    print("\nVerifying DataLoaders...")
    loaders = get_dataloaders(batch_size=32)
    for name, loader in loaders.items():
        images, labels = next(iter(loader))
        print(f"  {name} — batches: {len(loader)}, first batch shape: {tuple(images.shape)}")
