"""
run_benchmark_pretrained.py — Corruption sweep for the pretrained backbones.

Companion to run_benchmark.py. Trains ResNet18, ResNet50, and DenseNet121
across the same JPEG-corruption fractions [0%, 10%, 25%, 50%, 75%, 100%]
using a smaller learning rate (1e-4) suited to finetuning pretrained networks.

Outputs:
    - A summary table of best validation accuracy per model per fraction.
    - notebooks/03_corruption_sweep_pretrained.png — a plot that includes
      the toy CNN baseline so all four models can be compared at a glance.

Usage:
    python run_benchmark_pretrained.py

Prerequisites:
    - Data + split CSVs already in place. If not, run run_benchmark.py first
      (or call save_splits() from src/preprocessing.py).
"""

import copy
import io
import json
import random
import sys
import time
from pathlib import Path

# Force UTF-8 stdout so box-drawing / em-dash chars print correctly on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import (  # noqa: E402
    BrainTumorDataset,
    CLASS_NAMES,
    get_transform,
    load_split,
)
from model import get_model  # noqa: E402

SEED = 42
NUM_EPOCHS = 10
BATCH_SIZE = 32
FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
CORRUPTION = "jpeg"

# Smaller than the 1e-3 used in run_benchmark.py — these backbones already
# have good ImageNet features and don't need aggressive updates.
LR = 1e-4
WEIGHT_DECAY = 1e-4

MODEL_NAMES = ["resnet18", "resnet50", "densenet121"]

# Toy CNN baseline is loaded from outputs/toy_cnn_results.json, which is
# written at the end of run_benchmark.py. The toy CNN is NOT retrained here —
# run run_benchmark.py first to produce that file.
TOY_CNN_RESULTS_PATH = PROJECT_ROOT / "outputs" / "toy_cnn_results.json"


def load_toy_cnn_results():
    """Load the toy CNN best-val-acc baseline from disk.

    Returns a dict mapping corruption_fraction -> best_val_acc.
    Raises FileNotFoundError with a clear message if the file is missing.
    """
    if not TOY_CNN_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Toy CNN baseline not found at {TOY_CNN_RESULTS_PATH}.\n"
            f"  Run `python run_benchmark.py` first to generate it."
        )
    with open(TOY_CNN_RESULTS_PATH) as f:
        data = json.load(f)
    return dict(zip(data["fractions"], data["best_val_accs"]))


# ---------------------------------------------------------------------------
# Reproducibility & Device
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Corruption  (mirrors run_benchmark.py — keep behaviour identical)
# ---------------------------------------------------------------------------
def corrupt_jpeg(image, quality=10):
    """Apply heavy JPEG compression to *image*."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


CORRUPTIONS = {
    "jpeg": corrupt_jpeg,
}


# ---------------------------------------------------------------------------
# Corrupted Dataset  (mirrors run_benchmark.py)
# ---------------------------------------------------------------------------
class CorruptedBrainTumorDataset(BrainTumorDataset):
    """Wraps BrainTumorDataset and corrupts a configurable fraction of samples."""

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
    """Build a DataLoader where *corruption_fraction* of samples are corrupted."""
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


# ---------------------------------------------------------------------------
# Training  (mirrors run_benchmark.py — same loss/optimizer/scheduler pattern)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (avg loss, accuracy)."""
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
    """Evaluate model. Returns (avg loss, accuracy)."""
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


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------
def run_experiment(model_name, corruption_fraction, num_epochs, batch_size, device):
    """Train *model_name* fresh from ImageNet weights at the given corruption fraction.

    Validation is always evaluated on clean data. Returns a result dict.
    """
    print(f"\n  {'─'*56}")
    print(f"  Experiment: model={model_name}, corruption={CORRUPTION}, "
          f"fraction={corruption_fraction:.0%}")
    print(f"  {'─'*56}")

    # Corrupt training data; validate on clean data
    train_loader = build_corrupted_dataloader("train", CORRUPTION, corruption_fraction, batch_size)
    val_loader = build_corrupted_dataloader("val", CORRUPTION, 0.0, batch_size)

    # Fresh pretrained model with replaced final layer
    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
        "model_name": model_name,
        "corruption_fraction": corruption_fraction,
        "best_val_acc": best_val_acc,
        "history": history,
        "model_state": best_model_state,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(toy_cnn_results):
    """Train every model at every corruption fraction; print a summary table.

    *toy_cnn_results* is a dict mapping fraction -> best_val_acc, used only
    for the side-by-side comparison column in the summary table.
    """
    print("=" * 60)
    print("  Pretrained Corruption Benchmark")
    print("=" * 60)
    print(f"  Models:        {MODEL_NAMES}")
    print(f"  Fractions:     {FRACTIONS}")
    print(f"  Corruption:    {CORRUPTION}")
    print(f"  Epochs:        {NUM_EPOCHS}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"  Device:        {DEVICE}")

    # results[model_name] is a list of experiment dicts in the same order as FRACTIONS
    results = {name: [] for name in MODEL_NAMES}

    total_start = time.time()
    for model_name in MODEL_NAMES:
        print(f"\n\n##### Model: {model_name} #####")
        for frac in FRACTIONS:
            res = run_experiment(model_name, frac, NUM_EPOCHS, BATCH_SIZE, DEVICE)
            results[model_name].append(res)

    total_elapsed = time.time() - total_start
    print(f"\n  All experiments completed in {total_elapsed/60:.1f} minutes.")

    # --- Summary Table ---
    print("\n" + "=" * 60)
    print("  Summary: Best Validation Accuracy")
    print("=" * 60)
    col_w = 14
    header = f"  {'Fraction':<10}"
    for name in MODEL_NAMES:
        header += f"{name:<{col_w}}"
    header += f"{'toy_cnn*':<{col_w}}"
    print(header)
    print(f"  {'─' * (10 + col_w * (len(MODEL_NAMES) + 1))}")
    for i, frac in enumerate(FRACTIONS):
        row = f"  {frac:<10.0%}"
        for name in MODEL_NAMES:
            row += f"{results[name][i]['best_val_acc']:<{col_w}.4f}"
        row += f"{toy_cnn_results[frac]:<{col_w}.4f}"
        print(row)
    print(f"\n  * toy_cnn values loaded from {TOY_CNN_RESULTS_PATH.name} "
          "(written by run_benchmark.py).")

    return results


# ---------------------------------------------------------------------------
# Comparison Plot
# ---------------------------------------------------------------------------
def save_comparison_plot(results, toy_cnn_results):
    """Save notebooks/03_corruption_sweep_pretrained.png comparing all four models.

    *toy_cnn_results* is a dict mapping fraction -> best_val_acc, plotted as
    a dashed gray baseline alongside the three pretrained models.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not available — skipping plot generation.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left subplot: best val accuracy vs corruption fraction (all 4 models) ---
    for name in MODEL_NAMES:
        accs = [r["best_val_acc"] for r in results[name]]
        axes[0].plot(FRACTIONS, accs, "o-", linewidth=2, markersize=7, label=name)

    toy_accs = [toy_cnn_results[f] for f in FRACTIONS]
    axes[0].plot(
        FRACTIONS, toy_accs, "s--",
        linewidth=2, markersize=7, color="gray",
        label="toy_cnn (run_benchmark.py)",
    )

    axes[0].set_xlabel("Corruption Fraction")
    axes[0].set_ylabel("Best Validation Accuracy")
    axes[0].set_title(f"Val Accuracy vs. Training Corruption ({CORRUPTION})")
    axes[0].set_xticks(FRACTIONS)
    axes[0].set_xticklabels([f"{f:.0%}" for f in FRACTIONS])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # --- Right subplot: per-fraction val accuracy curves for resnet18 ---
    # (mirrors run_benchmark.py's second subplot; toy CNN per-epoch history
    # is not available here so we show the pretrained convergence pattern.)
    for r in results["resnet18"]:
        epochs = [h["epoch"] for h in r["history"]]
        vaccs = [h["val_acc"] for h in r["history"]]
        axes[1].plot(epochs, vaccs, label=f"{r['corruption_fraction']:.0%}")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("ResNet18 Training Curves by Corruption Fraction")
    axes[1].legend(title="Fraction")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "notebooks" / "03_corruption_sweep_pretrained.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the toy CNN baseline first so we fail fast if run_benchmark.py
    # hasn't been run yet — before spending hours training pretrained models.
    toy_cnn_results = load_toy_cnn_results()

    results = run_sweep(toy_cnn_results)
    save_comparison_plot(results, toy_cnn_results)

    print("\n" + "=" * 60)
    print("PRETRAINED BENCHMARK COMPLETE")
    print("=" * 60)
