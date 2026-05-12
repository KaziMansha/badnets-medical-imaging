"""
Pretrained model factory for the Brain Tumor MRI Dataset (Nickparvar, 4-class).

Provides ResNet18, ResNet50, and DenseNet121 with their final classification
layer replaced to output 4 classes (glioma, meningioma, notumor, pituitary).
"""

import torch.nn as nn
from torchvision import models

from preprocessing import CLASS_NAMES

#Constants
NUM_CLASSES = len(CLASS_NAMES)

#Model Builders
def _build_resnet18(num_classes):
    """ResNet18 with ImageNet weights and a new FC head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_resnet50(num_classes):
    """ResNet50 with ImageNet weights and a new FC head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_densenet121(num_classes):
    """DenseNet121 with ImageNet weights and a new classifier head."""
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


#Registry of available models
MODEL_REGISTRY = {
    "resnet18": _build_resnet18,
    "resnet50": _build_resnet50,
    "densenet121": _build_densenet121,
}


#Factory
def get_model(name, num_classes = NUM_CLASSES):
    """
    Look up *name* in the model registry and return the model with ImageNet
    weights loaded and the final classification layer replaced to output
    *num_classes* classes.

    Raises ``ValueError`` if the name is unknown.
    """
    builder = MODEL_REGISTRY.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Choose from: {sorted(MODEL_REGISTRY)}"
        )
    return builder(num_classes)


#Entry Point
if __name__ == "__main__":
    import torch

    from preprocessing import IMAGE_SIZE

    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    for name in MODEL_REGISTRY:
        model = get_model(name)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        print(f"{name:12s} -> output shape: {tuple(out.shape)}, "
              f"params: {sum(p.numel() for p in model.parameters()):,}")
