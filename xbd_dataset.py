# CELDA 54
"""
xBDDataset — Dataset de PyTorch para xBD (segmentación semántica de daños).

FIXES aplicados (M1):
  - DAMAGE_CLASSES empieza en 1 → background=0 es exclusivo y correcto.
  - _normalise usa normalización ImageNet estándar (sin 'eps' indefinido).
  - _build_seg_mask inicializa la máscara a 0 (background) en lugar de 255.
  - _print_class_distribution actualizado a 5 clases (0=background, 1-4=daños).
"""

import os, json, cv2, csv, random, time, copy
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import tifffile
from shapely.wkt import loads
from shapely.geometry import Polygon
from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm

# ─── Constantes ───────────────────────────────────────────────────────────────

# FIX: índices desplazados +1 para que background=0 sea exclusivo
DAMAGE_CLASSES = {
    "no-damage":    1,   # era 0 ← BUG CORREGIDO
    "minor-damage": 2,   # era 1
    "major-damage": 3,   # era 2
    "destroyed":    4,   # era 3
}
IDX_TO_CLASS = {v: k for k, v in DAMAGE_CLASSES.items()}
IDX_TO_CLASS[0] = "background"

TASKS      = {"classification", "detection", "segmentation"}
IMAGE_SIZE = 1024   # xBD siempre es 1024×1024

# Estadísticas ImageNet (rango [0,1]) — pasar con stats= al crear el Dataset
IMAGENET_STATS = {
    "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32),
    "std":  np.array([0.229, 0.224, 0.225], dtype=np.float32),
}


class xBDDataset(Dataset):
    """
    Parámetros
    ----------
    data_dir   : raíz del dataset xBD
    split      : lista de subdirectorios, ej. ['train'] o ['val','test']
    task       : 'classification' | 'segmentation' | 'detection'
    transform  : función (dict → dict) para data augmentation (solo train)
    patch_size : tamaño del patch cuadrado (px)
    stride     : radio de fusión para deduplicar ventanas
    stats      : dict {'mean': np.array([R,G,B]), 'std': np.array([R,G,B])}
                 en rango [0,1]. Usar IMAGENET_STATS para ResNet-101 pretrained.
    max_size   : límite de muestras (0 = sin límite; útil para debugging)
    """

    def __init__(
        self,
        data_dir:   str,
        split:      list = None,
        task:       str  = "classification",
        transform        = None,
        patch_size: int  = 64,
        stride:     int  = None,
        stats:      dict = None,
        max_size:   int  = 0,
    ):
        if split is None:
            split = ["train"]
        assert task in TASKS, f"task debe ser uno de {TASKS}, recibido '{task}'"

        self.data_dir   = data_dir
        self.split      = split
        self.task       = task
        self.transform  = transform
        self.max_size   = max_size
        self.patch_size = patch_size
        self.stride     = stride if stride is not None else patch_size
        self.stats      = stats

        self.image_pre_files  = []
        self.image_post_files = []
        self.label_pre_files  = []
        self.label_post_files = []

        for s in split:
            split_dir = os.path.join(data_dir, s)
            for d in sorted(os.listdir(split_dir)):
                img_dir = os.path.join(split_dir, d, "images")
                lbl_dir = os.path.join(split_dir, d, "labels")
                imgs = os.listdir(img_dir)
                lbls = os.listdir(lbl_dir)
                self.image_pre_files.append(
                    natsorted([os.path.join(img_dir, f) for f in imgs if "pre"  in f]))
                self.image_post_files.append(
                    natsorted([os.path.join(img_dir, f) for f in imgs if "post" in f]))
                self.label_pre_files.append(
                    natsorted([os.path.join(lbl_dir, f) for f in lbls if "pre"  in f]))
                self.label_post_files.append(
                    natsorted([os.path.join(lbl_dir, f) for f in lbls if "post" in f]))

        self.image_pre_files  = natsorted(np.hstack(self.image_pre_files))
        self.image_post_files = natsorted(np.hstack(self.image_post_files))
        self.label_pre_files  = natsorted(np.hstack(self.label_pre_files))
        self.label_post_files = natsorted(np.hstack(self.label_post_files))

        if self.max_size > 0:
            idx = np.random.RandomState(seed=42).permutation(
                range(len(self.image_pre_files)))[:self.max_size]
            self.image_pre_files  = np.array(self.image_pre_files)[idx]
            self.image_post_files = np.array(self.image_post_files)[idx]
            self.label_pre_files  = np.array(self.label_pre_files)[idx]
            self.label_post_files = np.array(self.label_post_files)[idx]

        self.samples = []
        self._build_samples()
        self._print_class_distribution()

    # ── Helpers internos ──────────────────────────────────────────────────────

    @staticmethod
    def _load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _read_image(path):
        img = tifffile.imread(path)
        if img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        return img

    def _parse_buildings(self, data_post):
        buildings = []
        for feat in data_post.get("features", {}).get("xy", []):
            props = feat.get("properties", {})
            if props.get("feature_type") != "building":
                continue
            wkt = feat.get("wkt", "")
            dmg = props.get("subtype", "no-damage")
            if dmg not in DAMAGE_CLASSES and dmg != -1:
                continue
            try:
                geom = loads(wkt)
                if not (isinstance(geom, Polygon) and geom.is_valid):
                    continue
                coords = np.array(geom.exterior.coords, dtype=np.float32)
                x1, y1 = coords.min(axis=0)
                x2, y2 = coords.max(axis=0)
                buildings.append({
                    "wkt":       wkt,
                    "label":     DAMAGE_CLASSES[dmg] if dmg in DAMAGE_CLASSES else -1,
                    "bbox_full": [float(x1), float(y1), float(x2), float(y2)],
                })
            except Exception:
                continue
        return buildings

    @staticmethod
    def _centred_window(cx, cy, ps):
        half = ps // 2
        x1 = int(np.clip(round(cx) - half, 0, IMAGE_SIZE - ps))
        y1 = int(np.clip(round(cy) - half, 0, IMAGE_SIZE - ps))
        return x1, y1, x1 + ps, y1 + ps

    @staticmethod
    def _dedup_windows(origins, stride):
        if stride == 0 or not origins:
            return origins
        accepted = []
        for ox, oy in origins:
            too_close = any(
                abs(ox - ax) <= stride and abs(oy - ay) <= stride
                for ax, ay in accepted
            )
            if not too_close:
                accepted.append((ox, oy))
        return accepted

    def _windows_for_image(self, buildings, ps):
        candidates = []
        for b in buildings:
            bx1, by1, bx2, by2 = b["bbox_full"]
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            x1, y1, _, _ = self._centred_window(cx, cy, ps)
            candidates.append((x1, y1))
        candidates.sort(key=lambda o: (o[1], o[0]))
        unique_origins = self._dedup_windows(candidates, self.stride)
        return [(x1, y1, x1 + ps, y1 + ps) for x1, y1 in unique_origins]

    def _build_samples(self):
        ps = min(self.patch_size, IMAGE_SIZE)
        for img_pre, img_post, lbl_pre, lbl_post in zip(
            self.image_pre_files, self.image_post_files,
            self.label_pre_files, self.label_post_files,
        ):
            data_post = self._load_json(lbl_post)
            buildings = self._parse_buildings(data_post)

            if self.patch_size >= IMAGE_SIZE:
                self.samples.append({
                    "img_pre":   img_pre,
                    "img_post":  img_post,
                    "lbl_post":  lbl_post,
                    "buildings": buildings,
                    "window":    (0, 0, IMAGE_SIZE, IMAGE_SIZE),
                })
                continue

            if self.task == "classification":
                for b in buildings:
                    bx1, by1, bx2, by2 = b["bbox_full"]
                    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    wx1, wy1, wx2, wy2 = self._centred_window(cx, cy, ps)
                    bbox_local = [
                        float(np.clip(bx1 - wx1, 0, ps)),
                        float(np.clip(by1 - wy1, 0, ps)),
                        float(np.clip(bx2 - wx1, 0, ps)),
                        float(np.clip(by2 - wy1, 0, ps)),
                    ]
                    self.samples.append({
                        "img_pre":   img_pre,
                        "img_post":  img_post,
                        "lbl_post":  lbl_post,
                        "buildings": [{**b, "bbox_local": bbox_local}],
                        "label":     b["label"],
                        "window":    (wx1, wy1, wx2, wy2),
                    })
                continue

            if not buildings:
                continue

            windows = self._windows_for_image(buildings, ps)
            for wx1, wy1, wx2, wy2 in windows:
                local_buildings = []
                for b in buildings:
                    bx1, by1, bx2, by2 = b["bbox_full"]
                    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    if wx1 <= cx < wx2 and wy1 <= cy < wy2:
                        local_buildings.append({
                            **b,
                            "bbox_local": [
                                float(np.clip(bx1 - wx1, 0, ps)),
                                float(np.clip(by1 - wy1, 0, ps)),
                                float(np.clip(bx2 - wx1, 0, ps)),
                                float(np.clip(by2 - wy1, 0, ps)),
                            ],
                        })
                self.samples.append({
                    "img_pre":   img_pre,
                    "img_post":  img_post,
                    "lbl_post":  lbl_post,
                    "buildings": local_buildings,
                    "window":    (wx1, wy1, wx2, wy2),
                })

    # ── I/O de imagen / patch ─────────────────────────────────────────────────

    def _crop_window(self, image, window):
        x1, y1, x2, y2 = window
        patch = image[y1:y2, x1:x2]
        ps = self.patch_size
        if patch.shape[:2] != (ps, ps):
            patch = cv2.resize(patch, (ps, ps))
        return patch

    def _build_seg_mask(self, image, sample):
        """
        Dibuja polígonos de edificios en una máscara de etiquetas.
        FIX: inicializa a 0 (background) en lugar de 255.
        Con DAMAGE_CLASSES={no-damage:1,...,destroyed:4}, los edificios
        se dibujan con valores 1-4 → nunca solapan con background (0).
        Mapa: 0=background, 1=no-damage, 2=minor-damage, 3=major-damage, 4=destroyed
        """
        ps                      = self.patch_size
        x1_win, y1_win, x2_win, y2_win = sample["window"]
        mask                    = np.zeros((ps, ps), dtype=np.uint8)  # FIX: 0=background
        scale                   = ps / (x2_win - x1_win)

        for b in sample["buildings"]:
            if b["label"] < 0:
                continue
            try:
                geom   = loads(b["wkt"])
                coords = np.array(geom.exterior.coords, dtype=np.float32)
                coords[:, 0] = (coords[:, 0] - x1_win) * scale
                coords[:, 1] = (coords[:, 1] - y1_win) * scale
                cv2.fillPoly(mask, [coords.astype(np.int32)], int(b["label"]))
            except Exception:
                continue
        return mask

    def _bboxes_for_sample(self, sample):
        boxes, labels = [], []
        for b in sample["buildings"]:
            if b["label"] < 0:
                continue
            boxes.append(b.get("bbox_local", b["bbox_full"]))
            labels.append(b["label"])
        if boxes:
            return (np.array(np.round(boxes), dtype=np.float32),
                    np.array(labels, dtype=np.int64))
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,),   dtype=np.int64))

    # ── Normalización ─────────────────────────────────────────────────────────

    def _normalise(self, patch: np.ndarray) -> np.ndarray:
        """
        FIX: Normalizacion ImageNet estandar (compatible con ResNet-101 pretrained).
        - Sin stats: escala a [0,1] dividiendo por 255.
        - Con stats : (patch/255 - mean) / std  ->  distribucion ~N(0,1) por canal.
        Elimina la variable eps que no estaba definida en la version original.
        """
        patch = patch.astype(np.float32) / 255.0   # siempre primero a [0,1]
        if self.stats is not None:
            mean  = np.reshape(self.stats["mean"], (1, 1, -1))
            std   = np.reshape(self.stats["std"],  (1, 1, -1))
            patch = (patch - mean) / (std + 1e-7)
        return patch

    # ── Estadísticas del dataset ──────────────────────────────────────────────

    def compute_statistics(self):
        """Calcula media y std por canal sobre todos los patches (en [0,1])."""
        print("Calculando estadísticas del dataset ...")
        total = np.zeros(3, dtype=np.float64)
        count = np.zeros(3, dtype=np.float64)
        for s in tqdm(self.samples):
            for path in (s["img_pre"], s["img_post"]):
                img   = self._read_image(path)
                patch = self._crop_window(img, s["window"]).astype(np.float64) / 255.0
                total += patch.sum(axis=(0, 1))
                count += patch.shape[0] * patch.shape[1]
        mean = total / count
        sq   = np.zeros(3, dtype=np.float64)
        for s in tqdm(self.samples):
            for path in (s["img_pre"], s["img_post"]):
                img   = self._read_image(path)
                patch = self._crop_window(img, s["window"]).astype(np.float64) / 255.0
                sq   += ((patch - mean) ** 2).sum(axis=(0, 1))
        std = np.sqrt(sq / count)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    # ── Diagnóstico ───────────────────────────────────────────────────────────

    def _print_class_distribution(self):
        """FIX: 5 clases correctas (0=background hasta 4=destroyed)."""
        counts = defaultdict(int)
        if self.task == "classification" and self.patch_size < IMAGE_SIZE:
            for s in self.samples:
                counts[s["label"]] += 1
        else:
            for s in self.samples:
                for b in s["buildings"]:
                    counts[b["label"]] += 1

        print(f"\n[xBDDataset] split={self.split}  task={self.task}  "
              f"patch_size={self.patch_size}  samples={len(self.samples)}")
        for cls, name in IDX_TO_CLASS.items():
            print(f"  {name:15s} (idx={cls}): {counts.get(cls, 0):>7d}")
        print(f"  unlabelled     (idx=-1): {counts.get(-1, 0):>7d}\n")

    # ── Protocolo Dataset ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_pre  = self._read_image(sample["img_pre"])
        img_post = self._read_image(sample["img_post"])

        patch_pre  = self._crop_window(img_pre,  sample["window"])
        patch_post = self._crop_window(img_post, sample["window"])

        # Máscara con valores 0-4 (sin 255 residuales)
        mask_patch = self._build_seg_mask(img_post, sample)

        # Normalizar y convertir a tensores CHW float
        patch_pre  = torch.from_numpy(
            np.transpose(self._normalise(patch_pre),  (2, 0, 1)))
        patch_post = torch.from_numpy(
            np.transpose(self._normalise(patch_post), (2, 0, 1)))
        mask_patch = torch.from_numpy(mask_patch.astype(np.int64))

        label_pre  = torch.tensor(0, dtype=torch.long)
        label_post = torch.tensor(
            sample.get("label", self.label_post_from_sample(sample)),
            dtype=torch.long)

        base = {
            "patch_pre_path":  sample["img_pre"],
            "patch_pre":       patch_pre,
            "patch_post_path": sample["img_post"],
            "patch_post":      patch_post,
            "mask_patch":      mask_patch,
            "label_pre":       label_pre,
            "label_post":      label_post,
            "idx":             idx,
        }

        if self.task == "detection":
            boxes, labels = self._bboxes_for_sample(sample)
            base["boxes"]  = torch.from_numpy(boxes)
            base["labels"] = torch.from_numpy(labels)

        # M1: el JointTransform se aplica aquí, solo si transform no es None
        if self.transform:
            base = self.transform(base)

        return base

    def label_post_from_sample(self, sample):
        labels = [b["label"] for b in sample["buildings"] if b["label"] >= 0]
        return max(labels) if labels else -1


# ─── Balanceo de clases compartido (baseline) ─────────────────────────────────

def _dominant_label(sample: dict) -> int:
    """Clase de daño más grave presente en el patch (0 si solo background)."""
    labels = [b["label"] for b in sample.get("buildings", []) if b["label"] > 0]
    return max(labels) if labels else 0


def balance_and_resplit(
    dataset_train: xBDDataset,
    dataset_val:   xBDDataset,
    val_size:      float = 0.15,
    random_state:  int   = 42,
):
    """
    Combina train + val y re-divide con estratificación por clase dominante.

    Replica exactamente la estrategia usada en el Proyecto 1 (celda 37):
      - all_samples  = train.samples + val.samples
      - split 85/15 estratificado por clase dominante del patch
      - random_state = 42 para reproducibilidad

    Parámetros
    ----------
    dataset_train : xBDDataset  (split=['train'])
    dataset_val   : xBDDataset  (split=['val'])
    val_size      : fracción de muestras para validación (default 0.15)
    random_state  : semilla para reproducibilidad (default 42)

    Devuelve
    --------
    (dataset_train_bal, dataset_val_bal) — dos xBDDataset con .samples actualizados.
    La configuración (transform, stats, patch_size, task) se hereda del original.

    Uso típico
    ----------
        from xbd_dataset import xBDDataset, IMAGENET_STATS, balance_and_resplit

        ds_train = xBDDataset(data_dir, split=['train'], ...)
        ds_val   = xBDDataset(data_dir, split=['val'],   ...)
        ds_train, ds_val = balance_and_resplit(ds_train, ds_val)
    """
    import copy
    import numpy as np
    from sklearn.model_selection import train_test_split

    # ── 1. Combinar muestras y calcular etiqueta dominante por muestra ────────
    all_samples = list(dataset_train.samples) + list(dataset_val.samples)
    all_labels  = np.array([_dominant_label(s) for s in all_samples])

    # ── 2. Split estratificado (misma proporción de clases en train y val) ────
    train_idx, val_idx = train_test_split(
        np.arange(len(all_samples)),
        test_size    = val_size,
        stratify     = all_labels,
        random_state = random_state,
    )

    # ── 3. Construir nuevos datasets reutilizando la configuración original ───
    ds_train_bal = copy.copy(dataset_train)
    ds_train_bal.samples = [all_samples[i] for i in train_idx]

    ds_val_bal = copy.copy(dataset_val)
    ds_val_bal.samples = [all_samples[i] for i in val_idx]

    # ── 4. Resumen ────────────────────────────────────────────────────────────
    _IDX_NAME = {0:"bg", 1:"no-dmg", 2:"minor", 3:"major", 4:"destr."}
    from collections import Counter
    orig_dist  = Counter(all_labels.tolist())
    train_dist = Counter([_dominant_label(s) for s in ds_train_bal.samples])
    val_dist   = Counter([_dominant_label(s) for s in ds_val_bal.samples])

    print("[balance_and_resplit] Split estratificado 85/15 por clase dominante:")
    print(f"  {'Clase':<10} {'Total':>7} {'Train':>7} {'Val':>7}")
    print(f"  {'-'*34}")
    for cls_id in sorted(orig_dist):
        print(f"  {_IDX_NAME.get(cls_id, str(cls_id)):<10} "
              f"{orig_dist[cls_id]:>7d} "
              f"{train_dist.get(cls_id, 0):>7d} "
              f"{val_dist.get(cls_id, 0):>7d}")
    print(f"  {'TOTAL':<10} {len(all_samples):>7d} "
          f"{len(ds_train_bal.samples):>7d} "
          f"{len(ds_val_bal.samples):>7d}")

    return ds_train_bal, ds_val_bal


def get_class_weights(dataset: xBDDataset, num_classes: int = 5) -> "torch.Tensor":
    """
    Calcula pesos de clase = total / (num_classes × count_por_clase).

    Replica la fórmula de P1 celda 41. Útil para CrossEntropy ponderado (M2)
    y para WeightedRandomSampler (baseline compartido).

    Parámetros
    ----------
    dataset     : xBDDataset  (tipicamente el de train)
    num_classes : número de clases incluyendo background (default 5)

    Devuelve
    --------
    torch.FloatTensor de shape (num_classes,) con los pesos por clase.
    """
    import numpy as np
    from collections import Counter

    # Contar clase dominante por muestra
    counts_dict = Counter([_dominant_label(s) for s in dataset.samples])
    total       = len(dataset.samples)

    counts_arr  = np.array(
        [counts_dict.get(i, 1) for i in range(num_classes)], dtype=np.float32
    )
    weights_np  = total / (num_classes * counts_arr)

    print("[get_class_weights] Pesos de clase (P1 fórmula):")
    _IDX_NAME = {0:"background", 1:"no-damage", 2:"minor-damage",
                 3:"major-damage", 4:"destroyed"}
    for i, (cnt, w) in enumerate(zip(counts_arr, weights_np)):
        print(f"  idx={i} {_IDX_NAME.get(i,'?'):<16} "
              f"muestras={int(cnt):>5d}  peso={w:.3f}")

    return torch.tensor(weights_np, dtype=torch.float32)