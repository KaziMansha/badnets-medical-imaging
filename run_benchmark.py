"""
run_benchmark.py — One-shot script to set up and run the full corruption benchmark.

Usage:
    python run_benchmark.py

What it does (in order):
    1. Installs Python dependencies from requirements.txt
    2. Downloads the Brain Tumor MRI dataset from Kaggle (requires kaggle CLI or manual download)
    3. Extracts data to data/raw/
    4. Runs preprocessing (generates train/val/test split CSVs)
    5. Trains a toy CNN at several corruption fractions and reports results
    6. Saves a summary plot to notebooks/02_corruption_sweep.png

Prerequisites:
    - Python 3.10+
    - (Optional) Kaggle API credentials for automatic download
      Set up via: pip install kaggle && kaggle configure
      Or place kaggle.json in ~/.kaggle/
    - If Kaggle CLI is unavailable, the script will prompt you to download manually.
"""

import copy
import io
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Force UTF-8 stdout so box-drawing / em-dash chars print correctly on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"

KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
KAGGLE_ZIP_NAME = "brain-tumor-mri-dataset.zip"

SEED = 42
NUM_EPOCHS = 10
BATCH_SIZE = 32
FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
CORRUPTION = "jpeg"  # Options: "jpeg", "gaussian_noise", "gaussian_blur"


# ---------------------------------------------------------------------------
# Step 1: Install dependencies
# ---------------------------------------------------------------------------
def install_dependencies():
    print("\n[1/5] Installing dependencies from requirements.txt...")
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        print("  WARNING: requirements.txt not found, skipping.")
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("  Done.")


# ---------------------------------------------------------------------------
# Step 2: Download & extract data
# ---------------------------------------------------------------------------
def download_and_extract_data():
    print("\n[2/5] Checking data...")

    # Check if data already exists
    training_dir = RAW_DIR / "Training"
    testing_dir = RAW_DIR / "Testing"
    if training_dir.is_dir() and testing_dir.is_dir():
        n_train = sum(1 for _ in training_dir.rglob("*") if _.is_file())
        n_test = sum(1 for _ in testing_dir.rglob("*") if _.is_file())
        if n_train > 0 and n_test > 0:
            print(f"  Data already exists ({n_train} train, {n_test} test files). Skipping download.")
            return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / KAGGLE_ZIP_NAME

    # Try Kaggle CLI
    if not zip_path.exists():
        print("  Attempting Kaggle CLI download...")
        try:
            subprocess.check_call(
                [
                    sys.executable, "-m", "kaggle", "datasets", "download",
                    "-d", KAGGLE_DATASET,
                    "-p", str(DATA_DIR),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("  Downloaded via Kaggle CLI.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  Kaggle CLI not available or not authenticated.")
            print()
            print("  ╔══════════════════════════════════════════════════════════════╗")
            print("  ║  MANUAL DOWNLOAD REQUIRED                                   ║")
            print("  ║                                                              ║")
            print("  ║  1. Go to:                                                   ║")
            print("  ║     https://www.kaggle.com/datasets/masoudnickparvar/        ║")
            print("  ║     brain-tumor-mri-dataset/download                         ║")
            print("  ║                                                              ║")
            print("  ║  2. Download the ZIP file                                    ║")
            print("  ║                                                              ║")
            print("  ║  3. Place it at:                                             ║")
            print(f"  ║     {zip_path}  ║")
            print("  ║                                                              ║")
            print("  ║  OR extract it directly so you have:                         ║")
            print(f"  ║     {RAW_DIR / 'Training'}           ║")
            print(f"  ║     {RAW_DIR / 'Testing'}            ║")
            print("  ║                                                              ║")
            print("  ╚══════════════════════════════════════════════════════════════╝")
            print()
            input("  Press Enter once the data is in place...")

            # Re-check
            if training_dir.is_dir() and testing_dir.is_dir():
                print("  Data directories found. Continuing.")
                return
            if not zip_path.exists():
                print("  ERROR: Neither ZIP nor extracted folders found. Exiting.")
                sys.exit(1)

    # Extract ZIP
    if zip_path.exists() and not (training_dir.is_dir() and testing_dir.is_dir()):
        print(f"  Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        print("  Extraction complete.")

        # Some Kaggle ZIPs nest an extra folder — fix if needed
        if not training_dir.is_dir():
            # Look one level deeper
            for child in RAW_DIR.iterdir():
                if child.is_dir() and (child / "Training").is_dir():
                    import shutil
                    for item in child.iterdir():
                        shutil.move(str(item), str(RAW_DIR / item.name))
                    child.rmdir()
                    break

    if not training_dir.is_dir() or not testing_dir.is_dir():
        print("  ERROR: Could not find Training/ and Testing/ directories after extraction.")
        sys.exit(1)

    print("  Data ready.")


# ---------------------------------------------------------------------------
# Step 3: Run preprocessing (generate split CSVs)
# ---------------------------------------------------------------------------
def run_preprocessing():
    print("\n[3/5] Running preprocessing (generating train/val/test splits)...")

    # Add src to path so we can import preprocessing
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from preprocessing import save_splits, SPLITS_DIR as _sd

    if (_sd / "train.csv").exists():
        print("  Split CSVs already exist. Skipping (delete data/splits/ to regenerate).")
    else:
        save_splits()
    print("  Done.")


# ---------------------------------------------------------------------------
# Step 4 & 5: Run the benchmark
# ---------------------------------------------------------------------------
def run_benchmark():
    print(f"\n[4/5] Running corruption benchmark (corruption={CORRUPTION}, epochs={NUM_EPOCHS})...")
    print(f"       Fractions: {FRACTIONS}")

    # Import everything we need
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from PIL import Image, ImageFilter
    from torch.utils.data import DataLoader

    from preprocessing import (
        BrainTumorDataset,
        CLASS_NAMES,
        get_transform,
        load_split,
    )

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"       Device: {DEVICE}")

    # --- Model ---
    class ToyCNN(nn.Module):
        def __init__(self, num_classes=4, in_channels=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # --- Corruptions ---
    def corrupt_jpeg(image, quality=10):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).copy()

    def corrupt_gaussian_noise(image, std=25.0):
        arr = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, std, arr.shape).astype(np.float32)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def corrupt_gaussian_blur(image, radius=4):
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    CORRUPTIONS = {
        "jpeg": corrupt_jpeg,
        "gaussian_noise": corrupt_gaussian_noise,
        "gaussian_blur": corrupt_gaussian_blur,
    }

    # --- Corrupted Dataset ---
    class CorruptedBrainTumorDataset(BrainTumorDataset):
        def __init__(self, samples, transform, corruption_fn, corruption_fraction=0.0, seed=42):
            super().__init__(samples, transform)
            self.corruption_fn = corruption_fn
            self.corruption_fraction = corruption_fraction
            n_corrupt = int(len(samples) * corruption_fraction)
            rng = random.Random(seed)
            all_indices = list(range(len(samples)))
            rng.shuffle(all_indices)
            self.corrupted_indices = set(all_indices[:n_corrupt])

        def __getitem__(self, idx):
            image_path, label = self.samples[idx]
            image = Image.open(image_path).convert("RGB")
            if idx in self.corrupted_indices:
                image = self.corruption_fn(image)
            if self.transform is not None:
                image = self.transform(image)
            return image, label

    def build_corrupted_dataloader(split, corruption_name, corruption_fraction, batch_size):
        samples = load_split(split)
        transform = get_transform(split)
        corruption_fn = CORRUPTIONS[corruption_name]
        dataset = CorruptedBrainTumorDataset(
            samples=samples,
            transform=transform,
            corruption_fn=corruption_fn,
            corruption_fraction=corruption_fraction,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)

    # --- Training ---
    def train_one_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return running_loss / total, correct / total

    @torch.no_grad()
    def evaluate(model, loader, criterion, device):
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return running_loss / total, correct / total

    # --- Experiment Runner ---
    def run_experiment(corruption_name, corruption_fraction, num_epochs, batch_size, device):
        print(f"\n  {'─'*56}")
        print(f"  Experiment: corruption={corruption_name}, fraction={corruption_fraction:.0%}")
        print(f"  {'─'*56}")

        train_loader = build_corrupted_dataloader("train", corruption_name, corruption_fraction, batch_size)
        val_loader = build_corrupted_dataloader("val", corruption_name, 0.0, batch_size)

        model = ToyCNN(num_classes=len(CLASS_NAMES)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        history = []
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0

            history.append({
                "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())

            if epoch % 2 == 0 or epoch == num_epochs:
                print(f"    Epoch {epoch:>3}/{num_epochs} — "
                      f"train_acc: {train_acc:.3f} | val_acc: {val_acc:.3f} | {elapsed:.1f}s")

        print(f"    Best val accuracy: {best_val_acc:.4f}")
        return {
            "corruption_name": corruption_name,
            "corruption_fraction": corruption_fraction,
            "best_val_acc": best_val_acc,
            "history": history,
            "model_state": best_model_state,
        }

    # --- Run Sweep ---
    results = []
    total_start = time.time()
    for frac in FRACTIONS:
        res = run_experiment(CORRUPTION, frac, NUM_EPOCHS, BATCH_SIZE, DEVICE)
        results.append(res)

    total_elapsed = time.time() - total_start
    print(f"\n  All experiments completed in {total_elapsed/60:.1f} minutes.")

    # --- Summary Table ---
    print(f"\n  {'Fraction':<12} {'Best Val Acc':<14}")
    print(f"  {'─'*26}")
    for r in results:
        print(f"  {r['corruption_fraction']:<12.0%} {r['best_val_acc']:<14.4f}")

    # --- Test Evaluation ---
    print("\n[5/5] Evaluating best model on clean & corrupted test sets...")
    best_result = max(results, key=lambda r: r["best_val_acc"])
    best_model = ToyCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    best_model.load_state_dict(best_result["model_state"])
    criterion = nn.CrossEntropyLoss()

    clean_test_loader = build_corrupted_dataloader("test", CORRUPTION, 0.0, BATCH_SIZE)
    _, clean_acc = evaluate(best_model, clean_test_loader, criterion, DEVICE)

    corrupted_test_loader = build_corrupted_dataloader("test", CORRUPTION, 1.0, BATCH_SIZE)
    _, corrupt_acc = evaluate(best_model, corrupted_test_loader, criterion, DEVICE)

    print(f"  Best model trained with fraction={best_result['corruption_fraction']:.0%}")
    print(f"  Clean test accuracy:     {clean_acc:.4f}")
    print(f"  Corrupted test accuracy: {corrupt_acc:.4f}")
    print(f"  Robustness gap:          {clean_acc - corrupt_acc:.4f}")

    # --- Save Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        fracs = [r["corruption_fraction"] for r in results]
        val_accs = [r["best_val_acc"] for r in results]

        axes[0].plot(fracs, val_accs, "o-", linewidth=2, markersize=8)
        axes[0].set_xlabel("Corruption Fraction")
        axes[0].set_ylabel("Best Validation Accuracy")
        axes[0].set_title(f"Val Accuracy vs. Training Corruption ({CORRUPTION})")
        axes[0].set_xticks(fracs)
        axes[0].set_xticklabels([f"{f:.0%}" for f in fracs])
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)

        for r in results:
            epochs = [h["epoch"] for h in r["history"]]
            vaccs = [h["val_acc"] for h in r["history"]]
            axes[1].plot(epochs, vaccs, label=f"{r['corruption_fraction']:.0%}")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Accuracy")
        axes[1].set_title("Training Curves by Corruption Fraction")
        axes[1].legend(title="Fraction")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plot_path = PROJECT_ROOT / "notebooks" / "02_corruption_sweep.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot saved to: {plot_path}")
    except ImportError:
        print("\n  matplotlib not available — skipping plot generation.")

    # --- Save Results to JSON ---
    # Persisted so downstream scripts (e.g. run_benchmark_pretrained.py) can
    # load the toy-CNN baseline without retraining it.
    results_path = PROJECT_ROOT / "outputs" / "toy_cnn_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_summary = {
        "corruption": CORRUPTION,
        "num_epochs": NUM_EPOCHS,
        "fractions": [r["corruption_fraction"] for r in results],
        "best_val_accs": [r["best_val_acc"] for r in results],
    }
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Brain Tumor MRI — Corruption Benchmark Runner")
    print("=" * 60)

    install_dependencies()
    download_and_extract_data()
    run_preprocessing()
    run_benchmark()
