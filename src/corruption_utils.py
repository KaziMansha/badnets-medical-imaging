"""
Image corruption utilities for robustness experiments.

Implements corruption methods discussed in the project README:
  - JPEG compression artifacts
  - Gaussian / Rician noise
  - Gibbs ringing (truncated k-space)
  - Rigid motion ghosting
  - Beam-hardening streaks (CT-style)
  - Stain / color-intensity variation
  - Tissue-fold simulation
  - Air-bubble simulation
"""

import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# 1. JPEG Compression
# ---------------------------------------------------------------------------

def jpeg_compress(image: Image.Image, quality: int = 10) -> Image.Image:
    """Re-encode *image* as JPEG at the given *quality* (1-95) and decode it
    back, introducing block-based compression artifacts."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# 2. Gaussian Noise
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    image: Image.Image,
    mean: float = 0.0,
    std: float = 25.0,
) -> Image.Image:
    """Add pixel-wise Gaussian noise with the given *mean* and *std*
    (on a 0-255 scale)."""
    arr = np.asarray(image, dtype=np.float32)
    rng = np.random.default_rng()
    noise = rng.normal(mean, std, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode=image.mode)


# ---------------------------------------------------------------------------
# 3. Rician Noise  (MRI-specific, signal-dependent)
# ---------------------------------------------------------------------------

def add_rician_noise(
    image: Image.Image,
    std: float = 25.0,
) -> Image.Image:
    """Add Rician noise (magnitude of complex Gaussian perturbation),
    which is the physically accurate noise model for MRI magnitude images."""
    arr = np.asarray(image, dtype=np.float64)
    rng = np.random.default_rng()
    real = arr + rng.normal(0, std, arr.shape)
    imag = rng.normal(0, std, arr.shape)
    noisy = np.sqrt(real ** 2 + imag ** 2)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode=image.mode)


# ---------------------------------------------------------------------------
# 4. Gibbs Ringing  (MRI k-space truncation)
# ---------------------------------------------------------------------------

def gibbs_ringing(
    image: Image.Image,
    truncation_fraction: float = 0.15,
) -> Image.Image:
    """Simulate Gibbs ringing by zeroing high-frequency components in
    k-space.  *truncation_fraction* (0, 1) controls how much of the outer
    k-space is removed — larger values produce stronger ringing."""
    arr = np.asarray(image, dtype=np.float64)

    def _truncate_channel(ch: np.ndarray) -> np.ndarray:
        kspace = np.fft.fft2(ch)
        kspace = np.fft.fftshift(kspace)
        rows, cols = ch.shape
        cr, cc = rows // 2, cols // 2
        r_cut = int(rows * (1 - truncation_fraction) / 2)
        c_cut = int(cols * (1 - truncation_fraction) / 2)
        mask = np.zeros_like(kspace)
        mask[cr - r_cut : cr + r_cut, cc - c_cut : cc + c_cut] = 1
        kspace *= mask
        kspace = np.fft.ifftshift(kspace)
        return np.abs(np.fft.ifft2(kspace))

    if arr.ndim == 2:
        result = _truncate_channel(arr)
    else:
        result = np.stack(
            [_truncate_channel(arr[:, :, c]) for c in range(arr.shape[2])],
            axis=-1,
        )

    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode=image.mode)


# ---------------------------------------------------------------------------
# 5. Rigid Motion Ghosting
# ---------------------------------------------------------------------------

def motion_ghosting(
    image: Image.Image,
    num_ghosts: int = 3,
    shift_range: Tuple[int, int] = (5, 15),
    intensity_decay: float = 0.5,
) -> Image.Image:
    """Simulate rigid-body motion ghosting by additively blending shifted,
    attenuated copies of the image.  Each successive ghost is dimmer by
    *intensity_decay*."""
    arr = np.asarray(image, dtype=np.float64)
    result = arr.copy()
    rng = np.random.default_rng()
    alpha = intensity_decay
    for _ in range(num_ghosts):
        dx = int(rng.integers(shift_range[0], shift_range[1] + 1))
        dy = int(rng.integers(shift_range[0], shift_range[1] + 1))
        dx *= rng.choice([-1, 1])
        dy *= rng.choice([-1, 1])
        shifted = np.roll(np.roll(arr, dx, axis=1), dy, axis=0)
        result = result + alpha * shifted
        alpha *= intensity_decay
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode=image.mode)


# ---------------------------------------------------------------------------
# 6. Beam-Hardening Streaks  (CT-style)
# ---------------------------------------------------------------------------

def beam_hardening(
    image: Image.Image,
    num_streaks: int = 5,
    streak_width: int = 3,
    darkness: float = 0.4,
) -> Image.Image:
    """Simulate CT beam-hardening by painting semi-transparent dark streaks
    across the image at random angles."""
    arr = np.asarray(image, dtype=np.float64)
    overlay = Image.new("L", image.size, 255)
    draw = ImageDraw.Draw(overlay)
    rng = np.random.default_rng()
    w, h = image.size
    for _ in range(num_streaks):
        angle = rng.uniform(0, np.pi)
        cx, cy = w / 2, h / 2
        length = max(w, h)
        x0 = int(cx - length * np.cos(angle) + rng.integers(-w // 4, w // 4))
        y0 = int(cy - length * np.sin(angle) + rng.integers(-h // 4, h // 4))
        x1 = int(cx + length * np.cos(angle) + rng.integers(-w // 4, w // 4))
        y1 = int(cy + length * np.sin(angle) + rng.integers(-h // 4, h // 4))
        draw.line([(x0, y0), (x1, y1)], fill=int(255 * darkness), width=streak_width)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=streak_width))
    overlay_arr = np.asarray(overlay, dtype=np.float64) / 255.0
    if arr.ndim == 3:
        overlay_arr = overlay_arr[:, :, np.newaxis]
    result = np.clip(arr * overlay_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode=image.mode)


# ---------------------------------------------------------------------------
# 7. Stain / Color-Intensity Variation  (Histopathology)
# ---------------------------------------------------------------------------

def stain_variation(
    image: Image.Image,
    hue_shift: float = 0.05,
    saturation_scale: float = 0.3,
    brightness_scale: float = 0.15,
) -> Image.Image:
    """Randomly shift hue, saturation, and brightness to mimic inter-lab
    stain variability in H&E pathology slides."""
    arr = np.asarray(image.convert("HSV"), dtype=np.float64)
    rng = np.random.default_rng()
    # Hue: channel 0 in [0, 255] for PIL HSV
    arr[:, :, 0] = (arr[:, :, 0] + rng.uniform(-hue_shift, hue_shift) * 255) % 256
    # Saturation
    s_factor = 1.0 + rng.uniform(-saturation_scale, saturation_scale)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * s_factor, 0, 255)
    # Value / brightness
    v_factor = 1.0 + rng.uniform(-brightness_scale, brightness_scale)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * v_factor, 0, 255)
    hsv_img = Image.fromarray(arr.astype(np.uint8), mode="HSV")
    return hsv_img.convert(image.mode)


# ---------------------------------------------------------------------------
# 8. Tissue-Fold Simulation  (Digital Pathology)
# ---------------------------------------------------------------------------

def tissue_fold(
    image: Image.Image,
    fold_width: int = 30,
    opacity_increase: float = 0.6,
) -> Image.Image:
    """Simulate a tissue fold by darkening a random band across the image,
    mimicking the doubled-thickness region caused by physical tissue folding."""
    arr = np.asarray(image, dtype=np.float64)
    h, w = arr.shape[:2]
    rng = np.random.default_rng()
    # Random horizontal or vertical band
    if rng.random() > 0.5:
        start = int(rng.integers(0, max(1, h - fold_width)))
        mask = np.ones_like(arr)
        mask[start : start + fold_width, :] = 1.0 - opacity_increase
        # Smooth edges of the fold
        mask_2d = mask[:, :, 0] if arr.ndim == 3 else mask
        mask_2d = gaussian_filter(mask_2d, sigma=fold_width / 4)
        if arr.ndim == 3:
            mask = np.stack([mask_2d] * arr.shape[2], axis=-1)
        else:
            mask = mask_2d
    else:
        start = int(rng.integers(0, max(1, w - fold_width)))
        mask = np.ones_like(arr)
        mask[:, start : start + fold_width] = 1.0 - opacity_increase
        mask_2d = mask[:, :, 0] if arr.ndim == 3 else mask
        mask_2d = gaussian_filter(mask_2d, sigma=fold_width / 4)
        if arr.ndim == 3:
            mask = np.stack([mask_2d] * arr.shape[2], axis=-1)
        else:
            mask = mask_2d
    result = np.clip(arr * mask, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode=image.mode)


# ---------------------------------------------------------------------------
# 9. Air-Bubble Simulation  (Digital Pathology)
# ---------------------------------------------------------------------------

def air_bubble(
    image: Image.Image,
    num_bubbles: int = 3,
    radius_range: Tuple[int, int] = (10, 40),
    brightness: float = 0.85,
) -> Image.Image:
    """Overlay bright, semi-transparent circles to simulate air bubbles
    trapped under a coverslip."""
    arr = np.asarray(image, dtype=np.float64)
    h, w = arr.shape[:2]
    rng = np.random.default_rng()
    overlay = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(overlay)
    for _ in range(num_bubbles):
        r = int(rng.integers(radius_range[0], radius_range[1] + 1))
        cx = int(rng.integers(r, w - r)) if w > 2 * r else w // 2
        cy = int(rng.integers(r, h - r)) if h > 2 * r else h // 2
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=255)
    bubble_mask = np.asarray(
        overlay.filter(ImageFilter.GaussianBlur(radius=5)),
        dtype=np.float64,
    ) / 255.0
    # Where the mask is active, push pixels toward white
    if arr.ndim == 3:
        bubble_mask = bubble_mask[:, :, np.newaxis]
    result = arr * (1 - bubble_mask) + 255 * brightness * bubble_mask
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode=image.mode)


# ---------------------------------------------------------------------------
# Convenience: apply a named corruption at a given severity
# ---------------------------------------------------------------------------

CORRUPTION_REGISTRY = {
    "jpeg_compression": jpeg_compress,
    "gaussian_noise": add_gaussian_noise,
    "rician_noise": add_rician_noise,
    "gibbs_ringing": gibbs_ringing,
    "motion_ghosting": motion_ghosting,
    "beam_hardening": beam_hardening,
    "stain_variation": stain_variation,
    "tissue_fold": tissue_fold,
    "air_bubble": air_bubble,
}


def apply_corruption(
    image: Image.Image,
    name: str,
    **kwargs,
) -> Image.Image:
    """Look up *name* in the corruption registry and apply it to *image*.

    Any extra keyword arguments are forwarded to the corruption function.
    Raises ``ValueError`` if the name is unknown.
    """
    fn = CORRUPTION_REGISTRY.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown corruption '{name}'. "
            f"Choose from: {sorted(CORRUPTION_REGISTRY)}"
        )
    return fn(image, **kwargs)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Default to a sample training image if no path is provided
    project_root = Path(__file__).resolve().parent.parent
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        img_path = next(
            (project_root / "data" / "raw" / "Training" / "glioma").glob("*.jpg"),
            None,
        )
        if img_path is None:
            print("No image found. Pass an image path as an argument.")
            sys.exit(1)

    print(f"Loading: {img_path}")
    original = Image.open(img_path).convert("RGB")

    out_dir = project_root / "corruption_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the original
    original_path = out_dir / f"{img_path.stem}_original.png"
    original.save(original_path)
    print(f"  Saved {'original':20s} -> {original_path.name}")

    # Build list of (label, image) for the comparison grid
    panels = [("Original", original)]

    # Apply every registered corruption and save the result
    for name in CORRUPTION_REGISTRY:
        corrupted = apply_corruption(original, name)
        out_path = out_dir / f"{img_path.stem}_{name}.png"
        corrupted.save(out_path)
        print(f"  Saved {name:20s} -> {out_path.name}")
        label = name.replace("_", " ").title()
        panels.append((label, corrupted))

    # --- Build a comparison grid ---
    from PIL import ImageFont

    thumb_size = 224
    padding = 10
    label_height = 24
    cols = 5
    rows = (len(panels) + cols - 1) // cols
    grid_w = cols * (thumb_size + padding) + padding
    grid_h = rows * (thumb_size + label_height + padding) + padding

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for idx, (label, img) in enumerate(panels):
        col = idx % cols
        row = idx // cols
        x = padding + col * (thumb_size + padding)
        y = padding + row * (thumb_size + label_height + padding)
        thumb = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        grid.paste(thumb, (x, y + label_height))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        tx = x + (thumb_size - tw) // 2
        draw.text((tx, y + 4), label, fill=(0, 0, 0), font=font)

    grid_path = out_dir / f"{img_path.stem}_comparison.png"
    grid.save(grid_path)
    print(f"\n  Comparison grid -> {grid_path.name}")
    print(f"All examples saved to {out_dir}")
