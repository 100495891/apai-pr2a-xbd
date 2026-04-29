"""
train_utils.py — Entrenamiento y evaluación de DeepLab-V3 sobre xBD.

Funciones expuestas
-------------------
- set_bn_eval                  : congela running stats de BatchNorm
- train_model_xbd              : bucle de entrenamiento (con hook aux para M3)
- test_segmentation_model_xbd  : evaluación con IoU por clase

Hooks abiertos
--------------
- M3: train_model_xbd acepta `use_aux=True, aux_weight=0.4` para sumar la pérdida
       de la cabeza auxiliar (outputs["aux"]) al total.

Origen
------
Adaptado del notebook de M1 (cells 35, 66, 67) sin cambios funcionales en el
camino del baseline (use_aux=False, criterion=CE → idéntico a M1).
"""

import os
import csv
import copy
import time
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_bn_eval(mm):
    """
    Pone BatchNorm en eval() para congelar running stats durante el entrenamiento.
    Importante con backbone preentrenado y batch pequeño.
    """
    if isinstance(mm, torch.nn.modules.batchnorm._BatchNorm):
        mm.eval()


def _denorm(tensor, stats=None):
    """Convierte tensor CHW normalizado a numpy HWC en [0,1] para visualización."""
    img = tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    if stats is not None:
        mean = np.array(stats["mean"], dtype=np.float32)
        std  = np.array(stats["std"],  dtype=np.float32)
        img  = img * std + mean
    return np.clip(img, 0.0, 1.0)


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def train_model_xbd(
    model, criterion, dataloaders, device,
    optimizer, lr_scheduler, metrics, bpath, model_name,
    num_classes: int = 5,
    num_epochs:  int = 3,
    use_aux:     bool  = False,           # ← M3
    aux_weight:  float = 0.4,             # ← M3
):
    """
    Bucle de entrenamiento adaptado para xBDDataset.

    Parámetros (los específicos)
    ----------------------------
    use_aux    : si True, suma la pérdida de outputs["aux"] al total
                 (requiere model construido con aux_classifier=True en arch.py).
    aux_weight : peso α de la pérdida auxiliar  (loss = main + α · aux).

    Resto: idéntico al baseline de M1.
    """
    since          = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_jaccard   = 0.0

    fieldnames = (["epoch", "Train_loss", "Val_loss"] +
                  [f"Train_{m}" for m in metrics] +
                  [f"Val_{m}"   for m in metrics])

    os.makedirs(bpath, exist_ok=True)
    with open(os.path.join(bpath, "log.csv"), "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for epoch in range(1, num_epochs + 1):
        ckpt_path = os.path.join(bpath, f"{model_name}-epoch{epoch}.pth")

        # Reanudación si ya existe checkpoint
        if os.path.exists(ckpt_path):
            print(f"> Cargando checkpoint epoca {epoch} ...")
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["scheduler"])
            best_jaccard = ckpt["best_jaccard"]
            continue

        print(f"\nEpoca {epoch}/{num_epochs}")
        print("-" * 40)
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ["Train", "Val"]:
            if phase == "Train":
                model.train()
            else:
                model.eval()
            # Congela running stats de BN del backbone preentrenado
            model.apply(set_bn_eval)

            for sample in tqdm(dataloaders[phase], desc=f"  {phase}"):
                inputs = sample["patch_post"].to(device)
                masks  = sample["mask_patch"].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "Train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs["out"], masks)

                    # Hook M3: aux_classifier
                    if use_aux and "aux" in outputs:
                        loss = loss + aux_weight * criterion(outputs["aux"], masks)

                    y_prob    = torch.nn.functional.softmax(outputs["out"], dim=1)
                    _, y_pred = torch.max(y_prob, dim=1)
                    y_pred_np = y_pred.data.cpu().numpy()
                    y_true_np = masks.data.cpu().numpy()

                    for name, metric in metrics.items():
                        if name == "jaccard_score":
                            # labels FIJOS = todas las clases, así ji[i] siempre es la clase i
                            ji = metric(y_true_np.ravel(), y_pred_np.ravel(),
                                        labels=list(range(num_classes)), average=None,
                                        zero_division=0)
                            # mIoU foreground = media de las 4 clases de daño (excluye background)
                            batchsummary[f"{phase}_{name}"].append(np.mean(ji[1:]))

                    if phase == "Train":
                        loss.backward()
                        optimizer.step()

            batchsummary["epoch"]         = epoch
            batchsummary[f"{phase}_loss"] = loss.item()
            print(f"  {phase} Loss: {loss.item():.4f}")

        for field in fieldnames[3:]:
            batchsummary[field] = np.nanmean(batchsummary[field])
        print(f"  Resumen: {batchsummary}")

        with open(os.path.join(bpath, "log.csv"), "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(batchsummary)

        val_iou = batchsummary["Val_jaccard_score"]
        if val_iou >= best_jaccard:
            best_jaccard   = val_iou
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({"state_dict": best_model_wts},
                       os.path.join(bpath, f"{model_name}_best.pth.tar"))
            print(f"  Nuevo mejor modelo guardado (Val mIoU = {best_jaccard:.4f})")

        lr_scheduler.step()
        torch.save({
            "epoch":        epoch,
            "state_dict":   model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    lr_scheduler.state_dict(),
            "best_jaccard": best_jaccard,
        }, ckpt_path)

    elapsed = time.time() - since
    print(f"\nEntrenamiento completo en {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"Mejor Val mIoU: {best_jaccard:.4f}")
    model.load_state_dict(best_model_wts)
    return model


# ─── Evaluación ───────────────────────────────────────────────────────────────

def test_segmentation_model_xbd(
    model, dataloaders, device, num_classes, class_names,
    result_dir, SAVE_OPT=True, batch_size=1, stats=None,
):
    """
    Evaluación de segmentación adaptada a xBDDataset.

    Devuelve
    --------
    cm_total        : matriz de confusión absoluta (num_classes × num_classes)
    iou_per_class   : array de IoU por clase de daño (excluye background) — útil
                      para que M2 cuantifique el efecto sobre clases minoritarias.
    """
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, "results_multiclass.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    if SAVE_OPT:
        os.makedirs(os.path.join(result_dir, "predictions_multiclass"), exist_ok=True)

    cmap_base   = plt.get_cmap("tab10" if num_classes <= 10 else "tab20")
    colors_list = [cmap_base(i) for i in range(num_classes)]
    colors_list[0] = (0, 0, 0, 1.0)
    custom_cmap = mcolors.ListedColormap(colors_list)
    norm_bins   = np.arange(num_classes + 1) - 0.5
    custom_norm = mcolors.BoundaryNorm(norm_bins, num_classes)

    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    jaccard  = []

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")

        model.eval()
        for sample in tqdm(dataloaders["Test"], desc="Evaluando"):
            with torch.no_grad():
                inputs = sample["patch_post"].to(device)
                masks  = sample["mask_patch"].to(device)

                y_prob    = torch.nn.functional.softmax(model(inputs)["out"], dim=1)
                _, y_pred = torch.max(y_prob, dim=1)
                y_pred_np = y_pred.data.cpu().numpy()
                y_true_np = masks.data.cpu().numpy()

            for j in range(y_pred_np.shape[0]):
                img_name = os.path.basename(sample["patch_post_path"][j])
                pred_j   = y_pred_np[j]
                true_j   = y_true_np[j]

                # IoU por clase (excluyendo background=0)
                ji = np.zeros(num_classes - 1, dtype=float)
                for i in range(1, num_classes):
                    pred_i       = pred_j == i
                    true_i       = true_j == i
                    intersection = np.logical_and(pred_i, true_i).sum()
                    union        = pred_i.sum() + true_i.sum() - intersection
                    ji[i - 1]    = intersection / union if union > 0 else 0.0
                jaccard.append(ji)

                cm_total += confusion_matrix(
                    true_j.ravel(), pred_j.ravel(),
                    labels=list(range(num_classes)),
                )

                if SAVE_OPT:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                                             constrained_layout=True)
                    axes[0].imshow(_denorm(inputs[j], stats))
                    axes[0].set_title("Post-desastre (entrada)", fontsize=12)
                    axes[0].axis("off")
                    axes[1].imshow(true_j, cmap=custom_cmap, norm=custom_norm,
                                   interpolation="nearest")
                    axes[1].set_title("Ground Truth", fontsize=12)
                    axes[1].axis("off")
                    axes[2].imshow(pred_j, cmap=custom_cmap, norm=custom_norm,
                                   interpolation="nearest")
                    axes[2].set_title("Prediccion", fontsize=12)
                    axes[2].axis("off")
                    save_path = os.path.join(
                        result_dir, "predictions_multiclass",
                        img_name[:-4] + ".png")
                    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                                pad_inches=0.1)
                    plt.close(fig)

                writer.writerow([img_name, str(ji)])

    iou_per_class = np.mean(jaccard, axis=0)

    print("\n── Resultados de evaluacion ───────────────────────")
    for i, name in enumerate(class_names[1:]):
        print(f"  IoU {name:15s}: {iou_per_class[i]:.4f}")
    print(f"  mIoU (fg, {num_classes-1} clases): {np.mean(iou_per_class):.4f}")
    print("───────────────────────────────────────────────────")

    with open(csv_path, "a", newline="") as csv_file:
        csv.writer(csv_file).writerow(["MEAN", str(iou_per_class)])

    # Matriz de confusión normalizada (%)
    fig, ax = plt.subplots(figsize=(10, 8))
    row_sums = cm_total.sum(axis=1)[:, np.newaxis]
    cm_norm  = np.divide(cm_total.astype(float), row_sums,
                         out=np.zeros_like(cm_total, dtype=float),
                         where=row_sums != 0)
    cm_pct   = np.round(cm_norm * 100.0, 2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_pct, display_labels=class_names)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax,
              xticks_rotation=45, text_kw={"fontsize": 9})
    plt.title("Matriz de Confusion Normalizada por Fila (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"), dpi=300)
    plt.show()

    return cm_total, iou_per_class
