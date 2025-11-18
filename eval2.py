import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.manifold import TSNE
from tqdm import tqdm
from datetime import datetime
import time
from stgcndual import LandmarkVideoDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
import re

def parse_training_log(log_path):
    # Parse lines, keep only first occurrence per epoch, and return every 10th epoch (10,20,...)
    data = {}
    pattern = re.compile(r"\[INFO\]\s*Epoch\s+(\d+)/\d+\s*-\s*loss\s+([0-9.eE+-]+)\s+acc\s+([0-9.eE+-]+)\s+val_acc\s+([0-9.eE+-]+)")
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            epoch = int(m.group(1))
            if epoch not in data:  # keep only the first occurrence for each epoch
                loss = float(m.group(2))
                acc = float(m.group(3))
                val_acc = float(m.group(4))
                data[epoch] = (loss, acc, val_acc)

    if not data:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # sort epochs and pick every 10th epoch (i.e., epochs divisible by 10)
    selected_epochs = sorted([e for e in data.keys() if e % 10 == 0])
    epochs = np.array(selected_epochs, dtype=int)
    loss = np.array([data[e][0] for e in selected_epochs], dtype=float)
    acc = np.array([data[e][1] for e in selected_epochs], dtype=float)
    intra_set_eval_acc = np.array([data[e][2] for e in selected_epochs], dtype=float)

    return epochs, loss, acc, intra_set_eval_acc

# ---------------------------
# Beispielaufrufe
# ---------------------------

# UNANONYMIZED DATASET
"""
python eval2.py \
  --checkpoint /netscratch/fschulz/didd_run01/best.pt \
  --data_root /netscratch/fschulz/didd_test_landmarks_npy \
  --save_dir /netscratch/fschulz/eval_run01_didd \
  --log_path /home/fschulz/video_deanon_train/configs/slurm-2221902.out \
  --match_root /netscratch/fschulz/test_landmarks_npy \
  --embedding_size 256
"""
# ANONYMIZED DATASET
"""
python eval2.py \
  --checkpoint /netscratch/fschulz/tavd_run05/best.pt \
  --data_root /netscratch/fschulz/test_landmarks_npy \
  --save_dir /netscratch/fschulz/eval_run05 \
  --log_path /home/fschulz/video_deanon_train/configs/slurm-2210858.out \
  --match_root /netscratch/fschulz/test_landmarks_npy \
  --embedding_size 256
"""
# UNANONYMIZED DATASET WITH CROSS-ENTROPY LOSS
"""
python eval2.py \
  --checkpoint /netscratch/fschulz/didd_run02_ce/best.pt \
  --data_root /netscratch/fschulz/didd_test_landmarks_npy \
  --save_dir /netscratch/fschulz/eval_run02_ce \
  --log_path /home/fschulz/video_deanon_train/configs/slurm-2238206.out \
  --match_root /netscratch/fschulz/test_landmarks_npy \
  --embedding_size 256
"""
# UNANONYMIZED DATASET WITH ARCFACE LOSS
"""
python eval2.py \
  --checkpoint /netscratch/fschulz/didd_run03_arcface/best.pt \
  --data_root /netscratch/fschulz/didd_test_landmarks_npy \
  --save_dir /netscratch/fschulz/eval_run03_arcface \
  --log_path /home/fschulz/video_deanon_train/configs/slurm-2238207.out \
  --match_root /netscratch/fschulz/test_landmarks_npy \
  --embedding_size 256
"""
# ANONYMIZED DATASET WITH CROSS-ENTROPY LOSS
"""
python eval2.py \
  --checkpoint /netscratch/fschulz/tavd_g6_ce/model/best.pt \
  --data_root /netscratch/fschulz/test_landmarks_npy \
  --save_dir /netscratch/fschulz/eval_tavd_g6_ce \
  --log_path /home/fschulz/video_deanon_train/configs/slurm-2258095.out \
  --match_root /netscratch/fschulz/test_landmarks_npy \
  --embedding_size 256
"""

# K-Folds
"""
python eval2.py \
  --checkpoint /netscratch/fschulz/tavd_k5_ce/best.pt \
  --data_root /netscratch/fschulz/cv_folds/k5_test_landmarks_npy \
  --save_dir /netscratch/fschulz/eval_tavd_k5_ce \
  --log_path /home/fschulz/video_deanon_train/configs/slurm-2258101.out \
  --embedding_size 256
"""


# ---------------------------
# Argumente
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Pfad zum Checkpoint (.pt)")
parser.add_argument("--data_root", type=str, required=True, help="Pfad zu Testdaten (.npy Dateien)")
parser.add_argument("--save_dir", type=str, default="./eval_results", help="Speicherort für Ergebnisse")
parser.add_argument("--log_path", type=str, default="./eval_results", help="Pfad zu Logdatei")
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--match_root", type=str, default=None, help="Optional: other dataset root to match files with (use to enforce identical file sets)")
start_time = time.time()
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

# ---------------------------
# Dummy-Modelstruktur importieren (wie beim Training)
# ---------------------------
from stgcndual import STGCNBackbone  # ggf. Pfad anpassen

# Create the dataset (set augmentation=False for deterministic sampling)
test_dataset = LandmarkVideoDataset(
    data_root=args.data_root,
    L=64,  # or the value used in training
    augmentation=False,
    input_scale=10.0,  # match training
    motion_scale=20.0, # match training
    seed=42,
    use_motion=False # optional: disable motion channels
)
# If requested, restrict test_dataset to files that also exist in args.match_root
if args.match_root:
    match_basenames = set()
    for root, _, files in os.walk(args.match_root):
        for f in files:
            if f.endswith(".npy"):
                match_basenames.add(f)
    orig_len = len(test_dataset.samples)
    # keep only samples whose basename exists in the other dataset
    test_dataset.samples = [(lbl, fp) for (lbl, fp) in test_dataset.samples if os.path.basename(fp) in match_basenames]
    new_len = len(test_dataset.samples)
    removed = orig_len - new_len
    print(f"🔹 match_root provided -> filtered test dataset: kept {new_len} / {orig_len} files (removed {removed})")
    if new_len == 0:
        raise RuntimeError(f"No overlapping .npy files found between {args.data_root} and {args.match_root}")

# ---------------------------
# Modell laden
# ---------------------------
device = torch.device(args.device)
sample_x, _ = test_dataset[0]
in_channels = sample_x.shape[0]
V = sample_x.shape[2]
num_layers = 9  # or the value you used during training
base_channels = 64  # or your training value
dropout = 0.3  # or your training value

model = STGCNBackbone(
    in_channels=in_channels,
    num_layers=num_layers,
    V=V,
    base_channels=base_channels,
    embedding_size=args.embedding_size,
    dropout=dropout
).to(device)
# model = STGCNBackbone(embedding_size=args.embedding_size).to(device)

print(f"🔹 Lade Checkpoint: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(ckpt["backbone"], strict=False)
model.eval()

# ---------------------------
# Testdaten laden
# ---------------------------
print(f"🔹 Lade Testdaten aus {args.data_root}")

embeddings, labels = [], []
for x, label in tqdm(test_dataset):
    x = x.unsqueeze(0).to(device)  # (1, C, L, V)
    with torch.no_grad():
        z = model(x)
        z = F.normalize(z, p=2, dim=1)
    embeddings.append(z.cpu())
    labels.append(label)

embeddings = torch.cat(embeddings)
unique_labels, label_indices = np.unique(labels, return_inverse=True)
labels = torch.tensor(label_indices)
print(f"✅ {len(embeddings)} Embeddings, {len(unique_labels)} Identitäten geladen")

# ---------------------------
# Paarbildung und Ähnlichkeiten
# ---------------------------
print("🔹 Berechne Paarähnlichkeiten...")
sims, targets = [], []
for i in tqdm(range(len(labels))):
    for j in range(i + 1, len(labels)):
        sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
        sims.append(sim)
        targets.append(int(labels[i] == labels[j]))

sims = np.array(sims)
targets = np.array(targets)

# ---------------------------
# ROC, AUC, EER
# ---------------------------
fpr, tpr, thresholds = roc_curve(targets, sims)
roc_auc = auc(fpr, tpr)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
thr_opt = thresholds[np.argmax(tpr - fpr)]
preds = (sims >= thr_opt).astype(int)
acc = accuracy_score(targets, preds)

# build balanced subset (same number of impostor pairs as genuine pairs)
same_idx = np.where(targets == 1)[0]
diff_idx = np.where(targets == 0)[0]
if len(diff_idx) >= len(same_idx) and len(same_idx) > 0:
    np.random.seed(42)
    diff_idx_sub = np.random.choice(diff_idx, size=len(same_idx), replace=False)
else:
    # fallback: sample with replacement if not enough negatives or no positives
    np.random.seed(42)
    diff_idx_sub = np.random.choice(diff_idx, size=max(1, len(same_idx)), replace=True)

balanced_idx = np.concatenate([same_idx, diff_idx_sub]) if len(same_idx) > 0 else diff_idx_sub
balanced_sims = sims[balanced_idx]
balanced_targets = targets[balanced_idx]

# metrics for balanced set
fpr_b, tpr_b, thresholds_b = roc_curve(balanced_targets, balanced_sims)
roc_auc_b = auc(fpr_b, tpr_b)
eer_b = fpr_b[np.nanargmin(np.abs(fpr_b - (1 - tpr_b)))]
thr_opt_b = thresholds_b[np.argmax(tpr_b - fpr_b)]
preds_b = (balanced_sims >= thr_opt_b).astype(int)
acc_b = accuracy_score(balanced_targets, preds_b)

elapsed = time.time() - start_time

# ---------------------------
# 📊 Summary (full + balanced)
summary_text = f"""
Evaluation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
======================================================
Checkpoint:     {args.checkpoint}
Data root:      {args.data_root}
Embeddings:     {len(embeddings)}
Identitäten:    {len(unique_labels)}

UNBALANCED (all pairs)
----------------------
ROC AUC:        {roc_auc:.4f}
EER:            {eer:.4f}
Accuracy:       {acc:.4f}
Optimal Thr:    {thr_opt:.4f}

BALANCED (subsampled impostors to match genuine count)
------------------------------------------------------
Kept pairs:     {len(balanced_sims)}
ROC AUC:        {roc_auc_b:.4f}
EER:            {eer_b:.4f}
Accuracy:       {acc_b:.4f}
Optimal Thr:    {thr_opt_b:.4f}

Laufzeit:       {elapsed/60:.2f} min
======================================================
"""
print(summary_text)
with open(os.path.join(args.save_dir, "eval_summary.txt"), "w") as f:
    f.write(summary_text)

# also save a compact json-like summary for easier programmatic use
try:
    import json
    compact = {
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "embeddings": len(embeddings),
        "n_identities": len(unique_labels),
        "unbalanced": {"roc_auc": float(roc_auc), "eer": float(eer), "acc": float(acc), "thr": float(thr_opt)},
        "balanced": {"roc_auc": float(roc_auc_b), "eer": float(eer_b), "acc": float(acc_b), "thr": float(thr_opt_b)}
    }
    with open(os.path.join(args.save_dir, "eval_summary_compact.json"), "w") as jf:
        json.dump(compact, jf, indent=2)
except Exception:
    pass

# ---------------------------
# Plots
# ---------------------------
print("🔹 Berechne ROC AUC...")
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "roc_curve.png"))
plt.close()

# Calculate FAR and FRR for all thresholds
FAR = fpr  # False Positive Rate is the same as False Acceptance Rate
FRR = 1 - tpr  # False Rejection Rate

print("🔹 Berechne EER...")
plt.figure(figsize=(6,6))
plt.plot(thresholds, FAR, label="False Acceptance Rate (FAR)")
plt.plot(thresholds, FRR, label="False Rejection Rate (FRR)")
plt.axvline(thr_opt, color="gray", linestyle="--", label=f"Optimal Threshold ({thr_opt:.3f})")
plt.axhline(eer, color="red", linestyle="--", label=f"EER ({eer:.3f})")
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("FAR & FRR vs. Threshold (EER Visualization)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "eer_plot.png"))
plt.close()

genuine_scores = sims[targets == 1]
impostor_scores = sims[targets == 0]

# Definiere FAR-Ziele (typische Werte für biometrische Szenarien)
far_targets = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
dir_values = []

for far in far_targets:
    # Threshold so bestimmen, dass gewählte FAR erreicht wird
    thr = np.percentile(impostor_scores, 100 * (1 - far))
    # Detection Identification Rate = TPR bei diesem Threshold
    dir_ = np.mean(genuine_scores >= thr)
    dir_values.append(dir_)

# ---- Plot ----
print("🔹 Berechne FAR DIR...")
plt.figure(figsize=(6,5))
plt.plot(far_targets, dir_values, marker='o', linewidth=2, label='DIR Curve')
plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.title('Detection Identification Rate (DIR) Curve')
plt.xlabel('False Accept Rate (FAR, log scale)')
plt.ylabel('Detection Identification Rate (DIR)')
plt.ylim(0, 1.05)
plt.xlim(max(far_targets), min(far_targets))  # Optional: x-Achse umdrehen (1e-1 → 1e-6)
plt.legend()

# Optional: Werte direkt im Plot anzeigen
for f, d in zip(far_targets, dir_values):
    plt.text(f, d + 0.02, f"{d:.2f}", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "dir_vs_far_curve.png"))
plt.close()

print("🔹 Berechne Cosine Similarity Histogramm...")
plt.figure(figsize=(6,4))
plt.hist(sims[targets==1], bins=50, alpha=0.6, label="Same identity", density=True)
plt.hist(sims[targets==0], bins=50, alpha=0.6, label="Different identity", density=True)
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.title("Cosine Similarity Distributions")
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "similarity_hist.png"))
plt.close()

print("🔹 Berechne t-SNE...")
tsne = TSNE(n_components=2, perplexity=20, random_state=42)
z_2d = tsne.fit_transform(embeddings)
plt.figure(figsize=(7,7))
plt.scatter(z_2d[:,0], z_2d[:,1], c=labels, cmap="tab20", s=8)
plt.title("t-SNE Embedding Space")
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "tsne_plot.png"))
plt.close()

print("🔹 Berechne Confusion Matrix...")
cm = confusion_matrix(targets, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix at Optimal Threshold")
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "confusion_matrix.png"))
plt.close()

print("🔹 Berechne Precision-Recall-Kurve...")
precision, recall, pr_thresholds = precision_recall_curve(targets, sims)
plt.figure(figsize=(6,6))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "precision_recall_curve.png"))
plt.close()

# print("🔹 Verteilung der Embedding-Normen...")
# norms = torch.norm(embeddings, p=2, dim=1).cpu().numpy()
# plt.figure(figsize=(6,4))
# plt.hist(norms, bins=50, alpha=0.7)
# plt.xlabel("Embedding Norm")
# plt.ylabel("Frequency")
# plt.title("Distribution of Embedding Norms")
# plt.tight_layout()
# plt.savefig(os.path.join(args.save_dir, "embedding_norms.png"))
# plt.close()

identity_similarities = []
for label in unique_labels:
    idx = (labels == label).nonzero(as_tuple=True)[0]
    sims_same = []
    for i in idx:
        for j in idx:
            if i < j:
                sims_same.append(F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item())
    identity_similarities.append(np.mean(sims_same) if sims_same else 0)
plt.figure(figsize=(10,4))
plt.bar(range(len(unique_labels)), identity_similarities)
plt.xlabel("Identity Index")
plt.ylabel("Mean Cosine Similarity (Same Identity)")
plt.title("Per-Identity Mean Similarity")
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "per_identity_similarity.png"))
plt.close()


# Subsample negative pairs for balanced plots
same_idx = np.where(targets == 1)[0]
diff_idx = np.where(targets == 0)[0]

# Randomly select as many negative pairs as there are positive pairs
np.random.seed(42)
diff_idx_sub = np.random.choice(diff_idx, size=len(same_idx), replace=False)

# Combine indices for balanced set
balanced_idx = np.concatenate([same_idx, diff_idx_sub])
balanced_sims = sims[balanced_idx]
balanced_targets = targets[balanced_idx]

# Confusion Matrix (balanced)
print("🔹 Berechne Confusion Matrix (balanced)...")
balanced_preds = (balanced_sims >= thr_opt).astype(int)
cm_balanced = confusion_matrix(balanced_targets, balanced_preds)
disp_balanced = ConfusionMatrixDisplay(confusion_matrix=cm_balanced, display_labels=["Different", "Same"])
disp_balanced.plot(cmap="Blues")
plt.title("Confusion Matrix (Balanced)")
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "confusion_matrix_balanced.png"))
plt.close()

# Precision-Recall Curve (balanced)
print("🔹 Berechne Precision-Recall-Kurve (balanced)...")
precision_bal, recall_bal, pr_thresholds_bal = precision_recall_curve(balanced_targets, balanced_sims)
plt.figure(figsize=(6,6))
plt.plot(recall_bal, precision_bal)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Balanced)")
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "precision_recall_curve_balanced.png"))
plt.close()

print("🔹 Berechne ROC AUC (balanced)...")
plt.figure(figsize=(6,6))
plt.plot(fpr_b, tpr_b, label=f"Balanced ROC (AUC = {roc_auc_b:.3f})", color="C3")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Balanced ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "roc_curve_balanced.png"))
plt.close()

# Calculate FAR and FRR for balanced thresholds
FAR_b = fpr_b
FRR_b = 1 - tpr_b

print("🔹 Berechne EER (balanced)...")
plt.figure(figsize=(6,6))
plt.plot(thresholds_b, FAR_b, label="Balanced FAR")
plt.plot(thresholds_b, FRR_b, label="Balanced FRR")
plt.axvline(thr_opt_b, color="gray", linestyle="--", label=f"Balanced Thr ({thr_opt_b:.3f})")
plt.axhline(eer_b, color="red", linestyle="--", label=f"Balanced EER ({eer_b:.3f})")
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("Balanced FAR & FRR vs. Threshold (EER Visualization)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "eer_plot_balanced.png"))
plt.close()

print("🔹 Berechne Cosine Similarity Histogramm (balanced)...")
plt.figure(figsize=(6,4))
plt.hist(balanced_sims[balanced_targets==1], bins=50, alpha=0.6, label="Same identity (balanced)", density=True)
plt.hist(balanced_sims[balanced_targets==0], bins=50, alpha=0.6, label="Different identity (balanced)", density=True)
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.legend()
plt.title("Cosine Similarity Distributions (Balanced)")
plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "similarity_hist_balanced.png"))
plt.close()

# DIR (balanced)
print("🔹 Berechne DIR Curve (balanced)...")
genuine_scores = balanced_sims[balanced_targets == 1]
impostor_scores = balanced_sims[balanced_targets == 0]

dir_values_balanced = []

for far in far_targets:
    # Threshold so bestimmen, dass gewählte FAR erreicht wird
    thr = np.percentile(impostor_scores, 100 * (1 - far))
    # Detection Identification Rate = TPR bei diesem Threshold
    dir_ = np.mean(genuine_scores >= thr)
    dir_values_balanced.append(dir_)

# ---- Plot ----
plt.figure(figsize=(6,5))
plt.plot(far_targets, dir_values_balanced, marker='o', linewidth=2, label='DIR Curve')
plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.title('Detection Identification Rate (DIR) Curve (Balanced)')
plt.xlabel('False Accept Rate (FAR, log scale)')
plt.ylabel('Detection Identification Rate (DIR)')
plt.ylim(0, 1.05)
plt.xlim(max(far_targets), min(far_targets))  # Optional: x-Achse umdrehen (1e-1 → 1e-6)
plt.legend()

# Optional: Werte direkt im Plot anzeigen
for f, d in zip(far_targets, dir_values_balanced):
    plt.text(f, d + 0.02, f"{d:.2f}", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "dir_vs_far_curve_balanced.png"))
plt.close()

# Plot training progress if log file exists
log_path = args.log_path
if os.path.exists(log_path):
    print("🔹 Lade Trainingsverlauf aus SLURM Log...")
    epochs, loss, acc, intra_set_eval_acc = parse_training_log(log_path)
    if len(epochs) > 0:
        # Loss plot
        plt.figure(figsize=(8,4))
        plt.plot(epochs, loss, label="Loss", color="C0")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "training_loss.png"))
        plt.close()

        # Accuracy plot (Train + Val)
        plt.figure(figsize=(8,4))
        plt.plot(epochs, acc, label="Train Acc", color="C1")
        plt.plot(epochs, intra_set_eval_acc, label="Intra Set Eval Acc", color="C2")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Intra Set Training Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "training_accuracy.png"))
        plt.close()
    else:
        print("⚠️ Keine Trainingsdaten im Log gefunden.")
else:
    print(f"⚠️ Trainingslog nicht gefunden: {log_path}")

print(f"✅ Evaluation abgeschlossen. Ergebnisse gespeichert unter: {args.save_dir}")