"""
augment.py — Data augmentation sincronizado para xBDDataset.

Modulos exportados:
- JointTransform         : transforma imagen y mascara de forma sincronizada.
- ABLATION_CONFIGS       : 5 configuraciones para el estudio de ablacion.
- visualize_augmentation : utilidad de verificacion visual imagen+mascara.
- compute_sample_weights : pesos para WeightedRandomSampler.
"""

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


class JointTransform:
    """
    Augmentation sincronizado para imagen y mascara.

    Aplica las mismas transformaciones espaciales (flip, rotacion, crop) a imagen
    y mascara, usando NEAREST para la mascara y BILINEAR para la imagen. Las
    transformaciones fotometricas (brillo, contraste, blur) se aplican solo a la imagen.

    Parametros
    ----------
    hflip_p          : probabilidad de flip horizontal          (default 0.5)
    vflip_p          : probabilidad de flip vertical            (default 0.5)
    rotation_degrees : rango de rotacion aleatoria en grados    (default 0.0)
    color_jitter_p   : probabilidad de aplicar brillo+contraste (default 0.0)
    brightness       : magnitud de variacion de brillo          (default 0.2)
    contrast         : magnitud de variacion de contraste       (default 0.2)
    blur_p           : probabilidad de aplicar Gaussian blur    (default 0.0)
    crop_p           : probabilidad de aplicar RandomResizedCrop (default 0.0)
    crop_scale_min   : fraccion minima del area a recortar      (default 0.5)
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
        crop_p: float = 0.0,
        crop_scale_min: float = 0.5,
    ):
        assert 0.0 <= hflip_p <= 1.0,        "hflip_p debe estar en [0,1]"
        assert 0.0 <= vflip_p <= 1.0,        "vflip_p debe estar en [0,1]"
        assert rotation_degrees >= 0,         "rotation_degrees debe ser >= 0"
        assert 0.0 <= color_jitter_p <= 1.0, "color_jitter_p debe estar en [0,1]"
        assert 0.0 <= blur_p <= 1.0,         "blur_p debe estar en [0,1]"
        assert 0.0 <= crop_p <= 1.0,         "crop_p debe estar en [0,1]"
        assert 0.0 < crop_scale_min <= 1.0,  "crop_scale_min debe estar en (0,1]"

        self.hflip_p          = hflip_p
        self.vflip_p          = vflip_p
        self.rotation_degrees = rotation_degrees
        self.color_jitter_p   = color_jitter_p
        self.brightness       = brightness
        self.contrast         = contrast
        self.blur_p           = blur_p
        self.crop_p           = crop_p
        self.crop_scale_min   = crop_scale_min

    def _random_crop_params(self, h: int, w: int):
        """Genera parametros de crop aleatorio (top, left, crop_h, crop_w)."""
        area = h * w
        for _ in range(10):  # hasta 10 intentos para encontrar crop válido
            scale       = random.uniform(self.crop_scale_min, 1.0)
            ratio       = random.uniform(3.0 / 4.0, 4.0 / 3.0)
            crop_w = int(round((area * scale * ratio) ** 0.5))
            crop_h = int(round((area * scale / ratio) ** 0.5))
            if 0 < crop_w <= w and 0 < crop_h <= h:
                top  = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                return top, left, crop_h, crop_w
        # Fallback: crop central del 80%
        crop_h = int(h * 0.8)
        crop_w = int(w * 0.8)
        top    = (h - crop_h) // 2
        left   = (w - crop_w) // 2
        return top, left, crop_h, crop_w

    def __call__(self, sample: dict) -> dict:
        img_post = sample["patch_post"]
        img_pre  = sample["patch_pre"]
        mask     = sample["mask_patch"]

        _, H, W = img_post.shape

        # Canal ficticio para la mascara (TF requiere dimension de canal)
        mask = mask.unsqueeze(0)

        # Transformaciones espaciales (sincronizadas imagen + mascara)

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

        # Crop aleatorio sincronizado (mismo recorte para imagen y mascara)
        if self.crop_p > 0 and random.random() < self.crop_p:
            top, left, crop_h, crop_w = self._random_crop_params(H, W)
            img_post = TF.resized_crop(img_post, top, left, crop_h, crop_w, (H, W),
                                       interpolation=TF.InterpolationMode.BILINEAR)
            img_pre  = TF.resized_crop(img_pre,  top, left, crop_h, crop_w, (H, W),
                                       interpolation=TF.InterpolationMode.BILINEAR)
            mask     = TF.resized_crop(mask.float(), top, left, crop_h, crop_w, (H, W),
                                       interpolation=TF.InterpolationMode.NEAREST).long()

        mask = mask.squeeze(0)

        # Transformaciones fotometricas (solo imagen, nunca la mascara)
        # Un unico check para brillo y contraste garantiza que se aplican juntos o ninguno.

        if random.random() < self.color_jitter_p:
            b = random.uniform(-self.brightness, self.brightness)
            c = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            img_post = (img_post + b) * c
            img_pre  = (img_pre  + b) * c

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
            f"color_jitter_p={self.color_jitter_p}, blur_p={self.blur_p}, "
            f"crop_p={self.crop_p}, crop_scale_min={self.crop_scale_min})"
        )


# Configuraciones de ablacion

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
    "aug_crop": {
        "transform":   JointTransform(
            hflip_p=0.5, vflip_p=0.5, rotation_degrees=15,
            color_jitter_p=0.5, brightness=0.2, contrast=0.2, blur_p=0.3,
            crop_p=0.5, crop_scale_min=0.5,
        ),
        "description": "Full Aug + RandomResizedCrop (escala 50-100%)",
    },
}


# Utilidades de visualizacion

CLASS_COLORS = {
    0: (0.0,  0.0,  0.0),   # background   (negro)
    1: (0.2,  0.8,  0.2),   # no-damage    (verde)
    2: (1.0,  0.85, 0.0),   # minor-damage (amarillo)
    3: (1.0,  0.4,  0.0),   # major-damage (naranja)
    4: (0.8,  0.0,  0.0),   # destroyed    (rojo)
}


def mask_to_rgb(mask_tensor):
    """Convierte una mascara (H,W) int64 a imagen RGB (H,W,3) para visualizacion."""
    mask_np = mask_tensor.cpu().numpy()
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.float32)
    for cls_idx, color in CLASS_COLORS.items():
        rgb[mask_np == cls_idx] = color
    return rgb


def denorm_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Desnormaliza un tensor ImageNet (C,H,W) a numpy (H,W,3) en rango [0,1]."""
    mean = np.array(mean).reshape(3, 1, 1)
    std  = np.array(std).reshape(3, 1, 1)
    return np.clip((tensor.cpu().numpy() * std + mean).transpose(1, 2, 0), 0, 1)


def compute_sample_weights(dataset) -> "torch.DoubleTensor":
    """
    Calcula un peso por muestra para WeightedRandomSampler.

    El peso de cada muestra es inversamente proporcional a la frecuencia de su
    clase dominante (el mayor nivel de dano presente en el patch). Devuelve un
    DoubleTensor de shape (len(dataset),) listo para WeightedRandomSampler.
    """
    from collections import Counter

    _IDX_TO_NAME = {
        0: "background",
        1: "no-damage",
        2: "minor-damage",
        3: "major-damage",
        4: "destroyed",
    }

    # Clase dominante por muestra (sin cargar imagenes)
    dominant = []
    for s in dataset.samples:
        labels = [b["label"] for b in s.get("buildings", []) if b["label"] > 0]
        dominant.append(max(labels) if labels else 0)

    # Frecuencia de cada clase dominante y calculo de pesos
    counts = Counter(dominant)
    total  = len(dominant)
    cls_w  = {cls: total / cnt for cls, cnt in counts.items()}

    weights = torch.DoubleTensor([cls_w[d] for d in dominant])

    print("[compute_sample_weights] Distribución de clases dominantes:")
    print(f"  {'idx':<4} {'nombre':<16} {'muestras':>9} {'peso_clase':>12}")
    print(f"  {'-'*45}")
    for cls_id in sorted(counts):
        name = _IDX_TO_NAME.get(cls_id, str(cls_id))
        print(f"  {cls_id:<4} {name:<16} {counts[cls_id]:>9d} {cls_w[cls_id]:>12.2f}x")
    print(f"  Total: {total} muestras")
    return weights


def visualize_augmentation(dataset, transform, n_samples=3, seed=42):
    """
    Muestra n_samples filas con 4 columnas: imagen original, imagen aumentada,
    mascara original y mascara aumentada. Permite verificar la coherencia espacial
    entre imagen y mascara tras aplicar el transform.
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
