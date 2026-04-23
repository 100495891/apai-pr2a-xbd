"""
augment.py — Data augmentation sincronizado para xBDDataset (Bloque M1).

Contenido
---------
- JointTransform     : transforma imagen + máscara de forma sincronizada
- ABLATION_CONFIGS   : 4 configuraciones predefinidas para la ablación de M1
- visualize_augmentation : utilidad para verificar visualmente que img+mask
                           se transforman de forma coherente

Uso típico
----------
    from augment import JointTransform, ABLATION_CONFIGS

    dataset_train = xBDDataset(
        data_dir=..., split=["train"], task="segmentation",
        patch_size=img_size, stats=IMAGENET_STATS,
        transform=JointTransform(hflip_p=0.5, vflip_p=0.5),
    )

Reglas de oro (M1)
------------------
- Transforms ESPACIALES (flip, rotación) → se aplican IGUAL a imagen Y máscara.
- Transforms FOTOMÉTRICOS (brillo, contraste, blur) → SOLO a la imagen.
- Rotación de máscara con NEAREST (preserva enteros 0-4); imágenes con BILINEAR.
"""

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


# ─── JointTransform ───────────────────────────────────────────────────────────

class JointTransform:
    """
    Data augmentation sincronizado imagen + máscara para xBDDataset.

    Recibe el dict de __getitem__ (tensores ya normalizados con ImageNet stats):
      - 'patch_post' : Tensor (C, H, W) float32  — imagen post-desastre
      - 'patch_pre'  : Tensor (C, H, W) float32  — imagen pre-desastre
      - 'mask_patch' : Tensor (H, W)    int64    — etiquetas 0-4

    Parámetros
    ----------
    hflip_p          : prob. flip horizontal              (default 0.5)
    vflip_p          : prob. flip vertical                (default 0.5)
    rotation_degrees : rango rotación aleatoria en grados (default 0 = sin rotación)
    color_jitter_p   : prob. de aplicar jitter brillo+contraste (default 0.0)
    brightness       : magnitud variación de brillo en [-b, +b]  (default 0.2)
    contrast         : magnitud variación de contraste en [1-c, 1+c] (default 0.2)
    blur_p           : prob. de aplicar Gaussian blur     (default 0.0)
    """

    def __init__(
        self,
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        rotation_degrees: float = 0.0,
        color_jitter_p: float = 0.0,
        brightness: float = 0.2,
        contrast: float = 0.2,
        blur_p: float = 0.0,
    ):
        assert 0.0 <= hflip_p <= 1.0,        "hflip_p debe estar en [0,1]"
        assert 0.0 <= vflip_p <= 1.0,        "vflip_p debe estar en [0,1]"
        assert rotation_degrees >= 0,         "rotation_degrees debe ser >= 0"
        assert 0.0 <= color_jitter_p <= 1.0, "color_jitter_p debe estar en [0,1]"
        assert 0.0 <= blur_p <= 1.0,         "blur_p debe estar en [0,1]"

        self.hflip_p          = hflip_p
        self.vflip_p          = vflip_p
        self.rotation_degrees = rotation_degrees
        self.color_jitter_p   = color_jitter_p
        self.brightness       = brightness
        self.contrast         = contrast
        self.blur_p           = blur_p

    def __call__(self, sample: dict) -> dict:
        img_post = sample["patch_post"]
        img_pre  = sample["patch_pre"]
        mask     = sample["mask_patch"]

        # TF necesita (C, H, W) → canal ficticio para la máscara
        mask = mask.unsqueeze(0)

        # ── ESPACIALES (sincronizados imagen + máscara) ───────────────────────

        if random.random() < self.hflip_p:
            img_post = TF.hflip(img_post)
            img_pre  = TF.hflip(img_pre)
            mask     = TF.hflip(mask)

        if random.random() < self.vflip_p:
            img_post = TF.vflip(img_post)
            img_pre  = TF.vflip(img_pre)
            mask     = TF.vflip(mask)

        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            img_post = TF.rotate(img_post, angle,
                                 interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            img_pre  = TF.rotate(img_pre,  angle,
                                 interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask     = TF.rotate(mask.float(), angle,
                                 interpolation=TF.InterpolationMode.NEAREST,  fill=0).long()

        mask = mask.squeeze(0)

        # ── FOTOMÉTRICOS (SOLO imágenes, NUNCA la máscara) ────────────────────

        if random.random() < self.color_jitter_p:
            b = random.uniform(-self.brightness, self.brightness)
            img_post = img_post + b
            img_pre  = img_pre  + b

        if random.random() < self.color_jitter_p:
            c = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            img_post = img_post * c
            img_pre  = img_pre  * c

        if random.random() < self.blur_p:
            ks = random.choice([3, 5])
            img_post = TF.gaussian_blur(img_post, kernel_size=ks, sigma=(0.1, 2.0))
            img_pre  = TF.gaussian_blur(img_pre,  kernel_size=ks, sigma=(0.1, 2.0))

        sample["patch_post"] = img_post
        sample["patch_pre"]  = img_pre
        sample["mask_patch"] = mask
        return sample

    def __repr__(self):
        return (
            f"JointTransform(hflip_p={self.hflip_p}, vflip_p={self.vflip_p}, "
            f"rotation_degrees={self.rotation_degrees}, "
            f"color_jitter_p={self.color_jitter_p}, blur_p={self.blur_p})"
        )


# ─── Configuraciones de ablación M1 ───────────────────────────────────────────

ABLATION_CONFIGS = {
    "aug_none": {
        "transform":   None,
        "description": "Sin augmentation (baseline)",
    },
    "aug_flips": {
        "transform":   JointTransform(hflip_p=0.5, vflip_p=0.5),
        "description": "Flips H + V",
    },
    "aug_geo": {
        "transform":   JointTransform(hflip_p=0.5, vflip_p=0.5, rotation_degrees=15),
        "description": "Flips + Rotación ±15°",
    },
    "aug_full": {
        "transform":   JointTransform(
            hflip_p=0.5, vflip_p=0.5, rotation_degrees=15,
            color_jitter_p=0.5, brightness=0.2, contrast=0.2, blur_p=0.3,
        ),
        "description": "Full Aug (Flips + Rot + Color + Blur)",
    },
}


# ─── Visualización antes/después ──────────────────────────────────────────────

CLASS_COLORS = {
    0: (0.0,  0.0,  0.0),   # background    → negro
    1: (0.2,  0.8,  0.2),   # no-damage     → verde
    2: (1.0,  0.85, 0.0),   # minor-damage  → amarillo
    3: (1.0,  0.4,  0.0),   # major-damage  → naranja
    4: (0.8,  0.0,  0.0),   # destroyed     → rojo
}


def mask_to_rgb(mask_tensor):
    """Convierte máscara (H,W) int64 → imagen RGB (H,W,3) para visualización."""
    mask_np = mask_tensor.cpu().numpy()
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.float32)
    for cls_idx, color in CLASS_COLORS.items():
        rgb[mask_np == cls_idx] = color
    return rgb


def denorm_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Desnormaliza tensor ImageNet (C,H,W) → numpy (H,W,3) en [0,1]."""
    mean = np.array(mean).reshape(3, 1, 1)
    std  = np.array(std).reshape(3, 1, 1)
    return np.clip((tensor.cpu().numpy() * std + mean).transpose(1, 2, 0), 0, 1)


def visualize_augmentation(dataset, transform, n_samples=3, seed=42):
    """
    Muestra n_samples filas con 4 columnas:
      col 0: imagen post-desastre original
      col 1: imagen post-desastre aumentada
      col 2: máscara original
      col 3: máscara aumentada
    Las cols 0-1 y 2-3 deben tener la misma transformación espacial.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    titles = ["Post-desastre\n(original)", "Post-desastre\n(aumentada)",
              "Máscara\n(original)",      "Máscara\n(aumentada)"]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=11)

    for row in range(n_samples):
        idx = random.randrange(len(dataset))
        sample_orig = dataset[idx]
        # Copia profunda para no contaminar al aplicar el transform
        sample_aug = {
            "patch_post": sample_orig["patch_post"].clone(),
            "patch_pre":  sample_orig["patch_pre"].clone(),
            "mask_patch": sample_orig["mask_patch"].clone(),
        }
        if transform is not None:
            sample_aug = transform(sample_aug)

        axes[row][0].imshow(denorm_image(sample_orig["patch_post"]))
        axes[row][1].imshow(denorm_image(sample_aug["patch_post"]))
        axes[row][2].imshow(mask_to_rgb(sample_orig["mask_patch"]))
        axes[row][3].imshow(mask_to_rgb(sample_aug["mask_patch"]))
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    return fig
