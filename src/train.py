"""
Generic training loop for the Brain Tumor MRI Dataset (Nickparvar, 4-class).

Works with any classification model that takes (B, 3, 224, 224) inputs and
outputs (B, num_classes) logits — including the toy CNN from notebook 02 and
the pretrained backbones from model.py.
"""

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

#Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

#Constants
DEFAULT_LR = 1e-4  #Small default so pretrained backbones don't wash out their ImageNet weights
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_EPOCHS = 10


#Training Helpers
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average training loss."""
    model.train()
    running_loss = 0.0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (average loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


#Driver
def train_model(
    model,
    loaders,
    num_epochs = DEFAULT_NUM_EPOCHS,
    lr = DEFAULT_LR,
    weight_decay = DEFAULT_WEIGHT_DECAY,
    checkpoint_name = "model",
    device = None,
):
    """
    Train *model* on loaders["train"] and validate on loaders["val"] each epoch.

    Uses CrossEntropyLoss and Adam. Prints training loss and validation accuracy
    after every epoch. Saves the best validation-accuracy checkpoint to:

        outputs/checkpoints/best_<checkpoint_name>.pt

    Parameters
    ----------
    model : nn.Module
        Any classifier returning logits of shape (B, num_classes).
    loaders : dict
        Output of preprocessing.get_dataloaders() — must contain "train" and "val".
    num_epochs : int
    lr : float
        Learning rate. Defaults to 1e-4 (good for pretrained backbones).
    weight_decay : float
    checkpoint_name : str
        Used to name the saved checkpoint file (best_<checkpoint_name>.pt).
    device : torch.device or None
        Auto-selects CUDA if available when None.

    Returns
    -------
    dict with keys: best_val_acc, history, checkpoint_path
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    history = []
    best_val_acc = 0.0
    best_state = None

    print(f"Training '{checkpoint_name}' for {num_epochs} epochs on {device} (lr={lr})")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"  Epoch {epoch:>3}/{num_epochs} — "
              f"train_loss: {train_loss:.4f} | "
              f"val_acc: {val_acc:.3f} | "
              f"{elapsed:.1f}s")

    #Save best checkpoint
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / f"best_{checkpoint_name}.pt"
    torch.save(best_state, checkpoint_path)
    print(f"Best val accuracy: {best_val_acc:.4f}  —  saved to {checkpoint_path}")

    return {
        "best_val_acc": best_val_acc,
        "history": history,
        "checkpoint_path": checkpoint_path,
    }


#Entry Point
if __name__ == "__main__":
    from model import get_model
    from preprocessing import get_dataloaders

    loaders = get_dataloaders(batch_size=32)
    model = get_model("resnet18")
    train_model(model, loaders, num_epochs=3, checkpoint_name="resnet18_smoke")
