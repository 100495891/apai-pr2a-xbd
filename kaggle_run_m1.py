# =============================================================================
# kaggle_run_m1.py — Ejecución completa M1 (Lara) en Kaggle
# Proyecto APAI-PR2A | Segmentación Semántica xBD | Grupo 08
#
# INSTRUCCIONES DE USO:
#   Copia cada celda (bloque entre "# ── CELDA X ──" y el siguiente) en una
#   celda de código de Kaggle y ejecútalas EN ORDEN, una a una.
#
#   Tiempo estimado de ejecución total con GPU T4:
#     Celda 3 — Baseline (10 épocas)        : ~25 min
#     Celda 5 — Ablación (5 configs × 10 ep): ~120 min
#     Celda 7 — WRS exp. (2 configs × 10 ep): ~50 min
#     TOTAL                                 : ~3.5 horas
#
#   ⚠️  REQUISITO PREVIO: configurar Kaggle Secret "GH_TOKEN"
#       Kaggle notebook → Add-ons → Secrets → Add new secret
#       Name: GH_TOKEN | Value: tu GitHub Personal Access Token
# =============================================================================


# ── CELDA 0 — Matplotlib inline (primera celda del notebook) ──────────────────
# Pega esto solo si ejecutas celdas sueltas, NO en el notebook completo.
# %matplotlib inline


# =============================================================================
# ── CELDA 1 — Instalación de dependencias ────────────────────────────────────
# =============================================================================
# Pega en Kaggle como celda de código y ejecuta.

import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "tifffile", "natsort", "shapely"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sympy==1.13.1"], check=True)

import importlib
for pkg in ['tifffile', 'natsort', 'shapely', 'cv2', 'sklearn', 'tqdm']:
    try:
        importlib.import_module(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ FALTA {pkg}")
print("Dependencias verificadas.")


# =============================================================================
# ── CELDA 2 — Setup: clonar repo (usa Kaggle Secret GH_TOKEN) ────────────────
# =============================================================================

import os, sys, subprocess
from kaggle_secrets import UserSecretsClient

GH_USER   = '100495767'
GH_OWNER  = '100495891'
GH_EMAIL  = '100495767@alumnos.uc3m.es'
BRANCH    = 'feat/augment'
REPO_PATH = '/kaggle/working/repo'

GH_TOKEN = UserSecretsClient().get_secret("GH_TOKEN")

REPO_URL_AUTH  = f'https://{GH_USER}:{GH_TOKEN}@github.com/{GH_OWNER}/apai-pr2a-xbd.git'
REPO_URL_CLEAN = f'https://github.com/{GH_OWNER}/apai-pr2a-xbd.git'

try:
    if not os.path.exists(REPO_PATH):
        subprocess.run(['git', 'clone', '--branch', BRANCH,
                        REPO_URL_AUTH, REPO_PATH], check=True)
    else:
        subprocess.run(['git', '-C', REPO_PATH, 'remote', 'set-url',
                        'origin', REPO_URL_AUTH], check=True)
        subprocess.run(['git', '-C', REPO_PATH, 'pull'], check=True)

    subprocess.run(['git', '-C', REPO_PATH, 'config',
                    'user.name',  GH_USER],  check=True)
    subprocess.run(['git', '-C', REPO_PATH, 'config',
                    'user.email', GH_EMAIL], check=True)

    sys.path.insert(0, REPO_PATH)
    os.chdir(REPO_PATH)

    branch = subprocess.check_output(
        ['git', '-C', REPO_PATH, 'branch', '--show-current']
    ).decode().strip()
    print(f'\n✅ Repo listo en {REPO_PATH}, rama: {branch}')
    print(f'   Python path: {sys.path[0]}')

finally:
    subprocess.run(['git', '-C', REPO_PATH, 'remote', 'set-url',
                    'origin', REPO_URL_CLEAN], check=False)
    GH_TOKEN = ''
    print('🔒 Token limpiado de memoria')


# =============================================================================
# ── CELDA 3 — Imports ─────────────────────────────────────────────────────────
# =============================================================================

import os
import glob
import json
import tifffile
from tqdm import tqdm
import numpy as np
import copy
import time
import torch
import torch.nn.functional as FT
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile, ImageDraw
import cv2
import csv
import random
import matplotlib
import matplotlib.pyplot as plt
from shapely.wkt import loads
from shapely.geometry import Polygon
from collections import defaultdict
from natsort import natsorted

# ── Módulos del proyecto ────────────────────────────────────────────────────
from xbd_dataset import xBDDataset, IMAGENET_STATS, DAMAGE_CLASSES, IDX_TO_CLASS
from augment     import JointTransform, ABLATION_CONFIGS, visualize_augmentation
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from losses      import get_loss
from arch        import get_deeplabv3_xbd
from train_utils import train_model_xbd, test_segmentation_model_xbd, set_bn_eval

# Semilla global de reproducibilidad
manualSeed = 999
print(f"Random Seed: {manualSeed}")
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.enabled = False

print("\nConfiguraciones de ablación M1 disponibles:")
for name, cfg in ABLATION_CONFIGS.items():
    print(f"  · {name:<10s} → {cfg['description']}")
print(f"\nxBDDataset disponible. Clases de daño: {DAMAGE_CLASSES}")
print("✅ Imports completados")


# =============================================================================
# ── CELDA 4 — Configuración (SMOKE_TEST=False, num_epochs=10) ─────────────────
# =============================================================================

DRIVE_BASE = '/kaggle/working'
data_dir   = '/kaggle/input/datasets/laragomezcarrera/mis-datos-vision/xBD_UC3M/xBD_UC3M'
result_dir = os.path.join(DRIVE_BASE, 'results_baseline')

num_workers     = 2
img_size        = 128
batchsize_train = 4
batchsize_test  = 1
num_classes     = 5

class_names = [
    'background',    # 0
    'no-damage',     # 1
    'minor-damage',  # 2
    'major-damage',  # 3
    'destroyed',     # 4
]

num_epochs  = 10
step_size   = 5     # LR decae ×0.1 en época 5 y 10
SMOKE_TEST  = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device    : {device}')
print(f'data_dir  : {data_dir}')
print(f'result_dir: {result_dir}')
print(f'Clases    : {num_classes} → {class_names}')
print(f'Epochs    : {num_epochs}  |  SMOKE_TEST: {SMOKE_TEST}')
assert not SMOKE_TEST, "⚠️ SMOKE_TEST debe ser False para el run final"


# =============================================================================
# ── CELDA 5 — Baseline compartido: re-split estratificado + WRS ───────────────
# =============================================================================

from collections import Counter

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base      = base_dataset
        self.indices   = np.array(indices)
        self.transform = transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        sample = self.base[self.indices[idx]]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

_IDX_NAME = {
    0: "background", 1: "no-damage",
    2: "minor-damage", 3: "major-damage", 4: "destroyed",
}

# 1. TEST separado (distribución real, sin balanceo)
dataset_test = xBDDataset(
    data_dir=data_dir, split=["test"], task="segmentation",
    patch_size=img_size, stats=IMAGENET_STATS, transform=None, max_size=0,
)

# 2. Pool combinado train+val
print("\nCargando splits train+val combinados ...")
dataset_all = xBDDataset(
    data_dir=data_dir, split=["train", "val"], task="segmentation",
    patch_size=img_size, stats=IMAGENET_STATS, transform=None, max_size=0,
)
print(f"  Total muestras combinadas: {len(dataset_all)}")

# 3. Clase dominante por muestra (la más grave, excluyendo background)
dominant_classes = np.array([
    max((b["label"] for b in s.get("buildings", []) if b["label"] > 0), default=0)
    for s in dataset_all.samples
])

print("\nDistribución ANTES del re-split:")
counts_all = Counter(dominant_classes.tolist())
total_all  = len(dominant_classes)
for cls_id in sorted(counts_all):
    n = counts_all[cls_id]
    print(f"  {_IDX_NAME.get(cls_id):15s}: {n:5d}  ({100*n/total_all:.1f}%)")

# 4. Re-split estratificado 85/15
all_indices = np.arange(len(dataset_all))
train_idx, val_idx = train_test_split(
    all_indices, test_size=0.15, stratify=dominant_classes, random_state=42,
)
print(f"\nRe-split 85/15 → Train: {len(train_idx)}  Val: {len(val_idx)}")

# 5. Subsets con transform de baseline
baseline_aug  = JointTransform(hflip_p=0.5, vflip_p=0.5)
dataset_train = TransformSubset(dataset_all, train_idx, transform=baseline_aug)
dataset_val   = TransformSubset(dataset_all, val_idx,   transform=None)

# 6. WeightedRandomSampler (fórmula Proyecto 1: w = total / num_cls / count)
train_dominant = dominant_classes[train_idx]
counts_train   = Counter(train_dominant.tolist())
total_train    = len(train_dominant)

cls_w = {cls: total_train / (num_classes * cnt)
         for cls, cnt in counts_train.items()}
sample_weights = torch.DoubleTensor([cls_w[d] for d in train_dominant])

print("\nPesos WeightedRandomSampler:")
for cls_id in sorted(cls_w):
    marker = "← submuestreado" if cls_w[cls_id] < 1 else "← sobremuestreado"
    print(f"  {_IDX_NAME.get(cls_id):15s}: {cls_w[cls_id]:.3f}×  {marker}")

wrs_sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True,
)

# 7. DataLoaders finales
dataloaders = {
    "Train": DataLoader(dataset_train, batch_size=batchsize_train,
                        sampler=wrs_sampler, num_workers=num_workers,
                        pin_memory=True),
    "Val":   DataLoader(dataset_val,   batch_size=batchsize_test,
                        shuffle=False,   num_workers=num_workers,
                        pin_memory=True),
    "Test":  DataLoader(dataset_test,  batch_size=batchsize_test,
                        shuffle=False,   num_workers=num_workers,
                        pin_memory=True),
}
# Para evaluación, usamos val como test (el split test de xBD no tiene labels)
dataloaders["Test"] = dataloaders["Val"]

print(f"\n✅ DataLoaders listos:")
print(f"  Train : {len(dataset_train):5d} muestras  (WRS activo)")
print(f"  Val   : {len(dataset_val):5d} muestras")
print(f"  Test  : {len(dataset_test):5d} muestras (sin labels — se usa Val)")


# =============================================================================
# ── CELDA 6 — Modelo baseline + Loss + Optimizer ─────────────────────────────
# =============================================================================

# Modelo DeepLabV3-ResNet101 adaptado a 5 clases xBD
model = get_deeplabv3_xbd(num_classes)
model.to(device)

# Loss: CrossEntropy estándar (baseline)
criterion = get_loss("cross_entropy")

# Optimizer: backbone con LR bajo (fine-tuning), classifier con LR alto
params_backbone   = [p for p in model.backbone.parameters()   if p.requires_grad]
params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]
optimizer = torch.optim.Adam([
    {"params": params_backbone},
    {"params": params_classifier, "lr": 1e-4},
], lr=1e-5)

# Scheduler: StepLR(step=5, gamma=0.1) → LR ×0.1 en épocas 5 y 10
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=step_size, gamma=0.1)

metrics = {"jaccard_score": jaccard_score}

print(f"Modelo  : DeepLabV3-ResNet101 ({num_classes} clases)")
print(f"Loss    : CrossEntropyLoss")
print(f"Opt     : Adam (backbone lr=1e-5, classifier lr=1e-4)")
print(f"Sched   : StepLR(step={step_size}, gamma=0.1)")
print("✅ Modelo listo")


# =============================================================================
# ── CELDA 7 — Entrenamiento baseline (10 épocas) ──────────────────────────────
# =============================================================================
import glob as _glob_mod

# Limpiar checkpoints previos (evita reanudaciones incorrectas)
if os.path.exists(result_dir):
    for _f in os.listdir(result_dir):
        if _f.endswith('.pth') or _f.endswith('.pth.tar'):
            os.remove(os.path.join(result_dir, _f))
            print(f'  Checkpoint anterior eliminado: {_f}')
os.makedirs(result_dir, exist_ok=True)

# Inicialización de pesos del clasificador (Xavier)
torch.manual_seed(manualSeed)
for m in model.classifier.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, 1.0)

trained_model = train_model_xbd(
    model, criterion, dataloaders, device,
    optimizer, lr_scheduler, metrics=metrics,
    bpath=result_dir, model_name="deeplabv3_baseline",
    num_classes=num_classes, num_epochs=num_epochs,
    use_aux=False,
)

# Limpiar epoch checkpoints (conservar solo _best)
for _f in _glob_mod.glob(os.path.join(result_dir, "*-epoch*.pth")):
    os.remove(_f)
print(f"\n✅ Baseline entrenado — checkpoints de época eliminados")


# =============================================================================
# ── CELDA 8 — Evaluación baseline ────────────────────────────────────────────
# =============================================================================

weights = torch.load(
    os.path.join(result_dir, "deeplabv3_baseline_best.pth.tar"),
    weights_only=False,
)["state_dict"]

model_eval = get_deeplabv3_xbd(num_classes)
model_eval.to(device)
model_eval.load_state_dict(weights)
model_eval.eval()

cm_baseline, iou_baseline = test_segmentation_model_xbd(
    model_eval, dataloaders, device,
    num_classes, class_names,
    result_dir, SAVE_OPT=True, stats=IMAGENET_STATS,
)

print("\n── RESULTADOS BASELINE ────────────────────────────────")
for i, name in enumerate(class_names[1:]):
    print(f"  IoU {name:15s}: {iou_baseline[i]:.4f}")
print(f"  mIoU (4 clases daño) : {np.mean(iou_baseline):.4f}")
print("──────────────────────────────────────────────────────")


# =============================================================================
# ── CELDA 9 — Ablación M1: 5 configuraciones de Data Augmentation ─────────────
# =============================================================================
import glob as _glob
import csv as _csv_abl

num_epochs_abl = num_epochs   # mismo num_epochs que baseline (10)
ablation_results = {}

for cfg_name, cfg in ABLATION_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"  Config: {cfg_name} — {cfg['description']}")
    print(f"{'='*60}")

    cfg_dir = os.path.join(DRIVE_BASE, 'results_m1', cfg_name)
    os.makedirs(cfg_dir, exist_ok=True)

    # Dataset train con el transform de esta config
    ds_train_cfg = xBDDataset(
        data_dir   = data_dir,
        split      = ['train'],
        task       = 'segmentation',
        patch_size = img_size,
        stats      = IMAGENET_STATS,
        transform  = cfg['transform'],
    )
    dl_train_cfg = DataLoader(
        ds_train_cfg, batch_size=batchsize_train,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    dataloaders_cfg = {
        'Train': dl_train_cfg,
        'Val':   dataloaders['Val'],
        'Test':  dataloaders['Val'],
    }

    # Modelo fresco con la misma semilla → comparación justa
    torch.manual_seed(manualSeed)
    model_cfg = get_deeplabv3_xbd(num_classes)
    model_cfg.to(device)
    for m in model_cfg.classifier.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight, 1.0)

    params_bb  = [p for p in model_cfg.backbone.parameters()   if p.requires_grad]
    params_cls = [p for p in model_cfg.classifier.parameters() if p.requires_grad]
    opt_cfg    = torch.optim.Adam(
        [{'params': params_bb}, {'params': params_cls, 'lr': 1e-4}], lr=1e-5)
    sched_cfg  = torch.optim.lr_scheduler.StepLR(
        opt_cfg, step_size=step_size, gamma=0.1)

    model_cfg = train_model_xbd(
        model_cfg, criterion, dataloaders_cfg, device,
        opt_cfg, sched_cfg,
        metrics     = {'jaccard_score': jaccard_score},
        bpath       = cfg_dir,
        model_name  = f'deeplabv3_{cfg_name}',
        num_classes = num_classes,
        num_epochs  = num_epochs_abl,
        use_aux     = False,
    )

    # Limpiar epoch checkpoints para no llenar el disco (solo conservar _best)
    for _f in _glob.glob(os.path.join(cfg_dir, "*-epoch*.pth")):
        os.remove(_f)

    # Evaluación
    print(f"  Evaluando {cfg_name}...")
    _, iou_per_class = test_segmentation_model_xbd(
        model_cfg, dataloaders_cfg, device, num_classes, class_names,
        result_dir=cfg_dir, SAVE_OPT=False, stats=IMAGENET_STATS,
    )

    # Mejor Val mIoU del log
    log_path = os.path.join(cfg_dir, 'log.csv')
    best_iou = 0.0
    if os.path.exists(log_path):
        with open(log_path) as lf:
            rows = list(_csv_abl.DictReader(lf))
        if rows:
            best_iou = max(float(r.get('Val_jaccard_score', 0)) for r in rows)

    ablation_results[cfg_name] = {
        'description':   cfg['description'],
        'best_val_iou':  best_iou,
        'iou_per_class': iou_per_class.tolist(),
    }
    print(f"  → Mejor Val mIoU: {best_iou:.4f}")
    for i, name in enumerate(class_names[1:]):
        print(f"     IoU {name:15s}: {iou_per_class[i]:.4f}")

# ── Tabla resumen ablación ────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"  M1 — ABLACIÓN: Comparativa Data Augmentation ({num_epochs_abl} épocas)")
print(f"{'='*100}")
print(f"  {'Config':<15} | {'Descripción':<38} | {'Val mIoU':>8} | "
      f"{'no-dmg':>7} | {'minor':>7} | {'major':>7} | {'destr.':>7}")
print(f"  {'-'*98}")
for n, res in ablation_results.items():
    ipc = res['iou_per_class']
    print(f"  {n:<15} | {res['description']:<38} | {res['best_val_iou']:>8.4f} | "
          f"{ipc[0]:>7.4f} | {ipc[1]:>7.4f} | {ipc[2]:>7.4f} | {ipc[3]:>7.4f}")
print(f"{'='*100}")
print("\n✅ Ablación M1 completada.")


# =============================================================================
# ── CELDA 10 — Gráficos comparativos de ablación ─────────────────────────────
# =============================================================================

import csv as _csv_plot

cfg_names    = list(ablation_results.keys())
descriptions = [ablation_results[n]['description'] for n in cfg_names]
val_ious     = [ablation_results[n]['best_val_iou'] for n in cfg_names]
bar_colors   = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#E68B29']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Bar chart Val mIoU por config
bars = axes[0].bar(descriptions, val_ious,
                   color=bar_colors[:len(cfg_names)], edgecolor='black', width=0.5)
y_max = max(val_ious) * 1.25 if max(val_ious) > 0 else 0.5
axes[0].set_ylim(0, y_max)
axes[0].set_ylabel('Val mIoU (excl. background)', fontsize=11)
axes[0].set_title('M1 — Comparativa Data Augmentation', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=20)
for bar, v in zip(bars, val_ious):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Gráfico 2: Curvas de pérdida por config
for color, cfg_name, desc in zip(bar_colors, cfg_names, descriptions):
    log_path = os.path.join(DRIVE_BASE, 'results_m1', cfg_name, 'log.csv')
    if not os.path.exists(log_path):
        continue
    with open(log_path) as lf:
        rows = list(_csv_plot.DictReader(lf))
    if not rows:
        continue
    epochs     = [int(r['epoch'])        for r in rows]
    train_loss = [float(r['Train_loss']) for r in rows]
    val_loss   = [float(r['Val_loss'])   for r in rows]
    axes[1].plot(epochs, train_loss, color=color, linestyle='-',
                 label=f'{desc} (train)', linewidth=1.5)
    axes[1].plot(epochs, val_loss,   color=color, linestyle='--',
                 label=f'{desc} (val)', linewidth=1.0)

axes[1].set_xlabel('Época', fontsize=11)
axes[1].set_ylabel('Loss (CrossEntropy)', fontsize=11)
axes[1].set_title('Curvas de entrenamiento por configuración', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=7, loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
os.makedirs(os.path.join(DRIVE_BASE, 'results_m1'), exist_ok=True)
fig_path = os.path.join(DRIVE_BASE, 'results_m1', 'M1_ablation_comparison.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Figura guardada en: {fig_path}')

# Tabla final con delta respecto al baseline (aug_none)
print(f'\n{"="*65}')
print(f"  {'Config':<15} | {'Descripción':<35} | {'Val mIoU':>8}")
print(f'{"="*65}')
baseline_iou = val_ious[0] if val_ious else 0.0
best_iou_val = max(val_ious)
for cfg_name, desc, iou in zip(cfg_names, descriptions, val_ious):
    delta     = iou - baseline_iou
    delta_str = f'(+{delta:.4f})' if delta > 0 else (f'({delta:.4f})' if delta < 0 else '   —   ')
    marker    = ' ← MEJOR' if iou == best_iou_val else ''
    print(f"  {cfg_name:<15} | {desc:<35} | {iou:>8.4f}  {delta_str}{marker}")
print(f'{"="*65}')


# =============================================================================
# ── CELDA 11 — Experimento final: aug_crop vs aug_crop + WRS ──────────────────
# =============================================================================
import glob as glob_module
import csv as _csv_wrs

num_epochs_wrs_exp = 10
wrs_experiment_results = {}

for exp_name, use_wrs in [
    ("aug_crop",     False),
    ("aug_crop_wrs", True),
]:
    print(f"\n{'='*60}")
    print(f"  Experimento : {exp_name}  (WRS = {'ON' if use_wrs else 'OFF'})")
    print(f"{'='*60}")

    exp_dir = os.path.join(DRIVE_BASE, 'results_m1', exp_name + '_final')
    os.makedirs(exp_dir, exist_ok=True)

    # Dataset train con aug_crop
    ds_exp = xBDDataset(
        data_dir=data_dir, split=['train'], task='segmentation',
        patch_size=img_size, stats=IMAGENET_STATS,
        transform=ABLATION_CONFIGS["aug_crop"]["transform"],
    )

    if use_wrs:
        # Cálculo inline de pesos WRS (fórmula P1)
        from collections import Counter as _Counter
        _dominant = [
            max((b["label"] for b in s.get("buildings", []) if b["label"] > 0),
                default=0)
            for s in ds_exp.samples
        ]
        _counts  = _Counter(_dominant)
        _total   = len(_dominant)
        _cls_w   = {cls: _total / (5 * cnt) for cls, cnt in _counts.items()}
        _weights = torch.DoubleTensor([_cls_w[d] for d in _dominant])
        exp_sampler = WeightedRandomSampler(
            _weights, len(_weights), replacement=True)
        dl_exp = DataLoader(ds_exp, batch_size=batchsize_train,
                            sampler=exp_sampler, num_workers=num_workers,
                            pin_memory=True)
    else:
        dl_exp = DataLoader(ds_exp, batch_size=batchsize_train,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True)

    dataloaders_exp = {
        "Train": dl_exp,
        "Val":   dataloaders["Val"],
        "Test":  dataloaders["Val"],
    }

    # Modelo fresco con semilla fija
    torch.manual_seed(manualSeed)
    model_exp = get_deeplabv3_xbd(num_classes)
    model_exp.to(device)
    for m in model_exp.classifier.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight, 1.0)

    params_bb  = [p for p in model_exp.backbone.parameters()   if p.requires_grad]
    params_cls = [p for p in model_exp.classifier.parameters() if p.requires_grad]
    opt_exp    = torch.optim.Adam(
        [{'params': params_bb}, {'params': params_cls, 'lr': 1e-4}], lr=1e-5)
    sched_exp  = torch.optim.lr_scheduler.StepLR(
        opt_exp, step_size=step_size, gamma=0.1)

    model_exp = train_model_xbd(
        model_exp, criterion, dataloaders_exp, device,
        opt_exp, sched_exp,
        metrics={'jaccard_score': jaccard_score},
        bpath=exp_dir,
        model_name=f'deeplabv3_{exp_name}',
        num_classes=num_classes,
        num_epochs=num_epochs_wrs_exp,
        use_aux=False,
    )

    # Limpiar epoch checkpoints
    for f in glob_module.glob(os.path.join(exp_dir, "*-epoch*.pth")):
        os.remove(f)

    # Evaluación
    print(f"\n  Evaluando modelo {exp_name}...")
    _, iou_per_class = test_segmentation_model_xbd(
        model_exp, dataloaders_exp, device,
        num_classes, class_names,
        exp_dir, SAVE_OPT=False, stats=IMAGENET_STATS,
    )

    # Mejor Val mIoU del log
    log_path = os.path.join(exp_dir, 'log.csv')
    best_val_iou = 0.0
    if os.path.exists(log_path):
        with open(log_path) as lf:
            rows = list(_csv_wrs.DictReader(lf))
        if rows:
            best_val_iou = max(float(r['Val_jaccard_score']) for r in rows)

    wrs_experiment_results[exp_name] = {
        "use_wrs":       use_wrs,
        "best_val_iou":  best_val_iou,
        "iou_per_class": iou_per_class,
    }
    print(f"\n  → Mejor Val mIoU: {best_val_iou:.4f}")

# ── Tabla comparativa aug_crop vs aug_crop+WRS ────────────────────────────────
print(f"\n{'='*80}")
print(f"  M1 — EXPERIMENTO FINAL: aug_crop vs aug_crop + WRS  ({num_epochs_wrs_exp} épocas)")
print(f"{'='*80}")
print(f"  {'Experimento':<20} {'Val mIoU':>8} {'no-dmg':>8} "
      f"{'minor':>8} {'major':>8} {'destr.':>8}  {'delta':>10}")
print(f"  {'-'*76}")
base_iou = None
for exp_name, res in wrs_experiment_results.items():
    ipc  = res['iou_per_class']
    viou = res['best_val_iou']
    if base_iou is None:
        delta_str = "  —"
        base_iou  = viou
    else:
        delta_str = f"(+{viou-base_iou:.4f})" if viou >= base_iou else f"({viou-base_iou:.4f})"
    marker = " ← WRS ON" if res["use_wrs"] else ""
    print(f"  {exp_name:<20} {viou:>8.4f} {ipc[0]:>8.4f} {ipc[1]:>8.4f} "
          f"{ipc[2]:>8.4f} {ipc[3]:>8.4f}  {delta_str}{marker}")
print(f"\n✅ M1 Experimento final completado.")


# =============================================================================
# ── CELDA 12 — RESUMEN FINAL M1 (todos los resultados clave) ──────────────────
# =============================================================================

print("\n" + "="*80)
print("  RESUMEN FINAL M1 — RESULTADOS PARA LA MEMORIA")
print("="*80)

# 1. Baseline
print("\n[1] BASELINE (WRS activo, sin augmentation adicional, 10 épocas)")
print(f"    mIoU (4 clases daño): {np.mean(iou_baseline):.4f}")
for i, name in enumerate(class_names[1:]):
    print(f"    IoU {name:15s}: {iou_baseline[i]:.4f}")

# 2. Ablación
print("\n[2] ABLACIÓN — 5 configuraciones de Data Augmentation (10 épocas c/u)")
print(f"  {'Config':<15} | {'Val mIoU':>8} | {'no-dmg':>7} | "
      f"{'minor':>7} | {'major':>7} | {'destr.':>7}")
print(f"  {'-'*62}")
best_cfg = max(ablation_results, key=lambda k: ablation_results[k]['best_val_iou'])
for n, res in ablation_results.items():
    ipc    = res['iou_per_class']
    marker = " ← MEJOR" if n == best_cfg else ""
    print(f"  {n:<15} | {res['best_val_iou']:>8.4f} | {ipc[0]:>7.4f} | "
          f"{ipc[1]:>7.4f} | {ipc[2]:>7.4f} | {ipc[3]:>7.4f}{marker}")

# 3. WRS experiment
print("\n[3] EXPERIMENTO WRS — aug_crop sin WRS vs aug_crop + WRS (10 épocas c/u)")
print(f"  {'Experimento':<20} | {'Val mIoU':>8} | {'no-dmg':>7} | "
      f"{'minor':>7} | {'major':>7} | {'destr.':>7}")
print(f"  {'-'*68}")
for exp_name, res in wrs_experiment_results.items():
    ipc    = res['iou_per_class']
    viou   = res['best_val_iou']
    marker = " ← WRS ON" if res["use_wrs"] else ""
    print(f"  {exp_name:<20} | {viou:>8.4f} | {ipc[0]:>8.4f} | "
          f"{ipc[1]:>8.4f} | {ipc[2]:>8.4f} | {ipc[3]:>8.4f}{marker}")

print("\n" + "="*80)
print("  Copia este bloque en tu documento de resultados (RESULTADOS.docx)")
print("="*80)
