import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from stgcndual import STGCNBackbone, LandmarkVideoDataset

def get_roc_from_saved(dirpath):
    fpr_p = os.path.join(dirpath, "roc_fpr.npy")
    tpr_p = os.path.join(dirpath, "roc_tpr.npy")
    if os.path.exists(fpr_p) and os.path.exists(tpr_p):
        fpr = np.load(fpr_p)
        tpr = np.load(tpr_p)
        roc_auc = float(auc(fpr, tpr))
        return fpr, tpr, roc_auc
    return None

def compute_embeddings_and_roc(checkpoint, data_root, embedding_size, device, use_motion):
    # build dataset
    ds = LandmarkVideoDataset(data_root=data_root, L=64, augmentation=False, input_scale=10.0, motion_scale=20.0, seed=42, use_motion=use_motion)
    if len(ds.samples) == 0:
        raise RuntimeError(f"No samples found in {data_root}")
    sample_x, _ = ds[0]
    in_channels = sample_x.shape[0]
    V = sample_x.shape[2]

    model = STGCNBackbone(in_channels=in_channels, num_layers=9, V=V, base_channels=64, embedding_size=embedding_size, dropout=0.3).to(device)
    ck = torch.load(checkpoint, map_location=device)
    # try to load backbone dict fallback
    try:
        model.load_state_dict(ck.get("backbone", ck), strict=False)
    except Exception:
        model.load_state_dict(ck, strict=False)
    model.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for x, lbl in ds:
            x = x.unsqueeze(0).to(device)
            z = model(x)
            z = F.normalize(z, p=2, dim=1)
            embeddings.append(z.cpu().numpy())
            labels.append(int(lbl))
    embeddings = np.vstack(embeddings)  # (N, D)
    labels = np.array(labels, dtype=int)
    # pairwise sims
    sim_mat = embeddings @ embeddings.T
    iu = np.triu_indices(len(labels), k=1)
    sims = sim_mat[iu]
    targets = (labels[iu[0]] == labels[iu[1]]).astype(int)
    if targets.sum() == 0 or (targets == 0).sum() == 0:
        raise RuntimeError("Not enough positive or negative pairs to compute ROC")
    fpr, tpr, _ = roc_curve(targets, sims)
    roc_auc = float(auc(fpr, tpr))
    return fpr, tpr, roc_auc

# K-Folds
"""
python plot_rocs2.py \
  --checkpoints /netscratch/fschulz/tavd_k5_ce/best.pt \
  /netscratch/fschulz/tavd_k4_ce/best.pt \
  /netscratch/fschulz/tavd_k3_ce/best.pt \
  /netscratch/fschulz/tavd_k2_ce/best.pt \
  /netscratch/fschulz/tavd_k1_ce/best.pt \
  /netscratch/fschulz/tavd_xy_ce/best.pt \
  --data_root /netscratch/fschulz/cv_folds/k5_test_landmarks_npy \
  /netscratch/fschulz/cv_folds/k4_test_landmarks_npy \
  /netscratch/fschulz/cv_folds/k3_test_landmarks_npy \
  /netscratch/fschulz/cv_folds/k2_test_landmarks_npy \
  /netscratch/fschulz/cv_folds/k1_test_landmarks_npy \
  /netscratch/fschulz/test_landmarks_npy \
  --save_path /netscratch/fschulz/plot_rocs_k.png \
  --labels TAVD_K5+TAVD_K4+TAVD_K3+TAVD_K2+TAVD_K1+TAVD_Baseline \
  --embedding_size 256
"""

# Groups
"""
python plot_rocs2.py \
  --checkpoints /netscratch/fschulz/tavd_g6_ce/model/best.pt \
  /netscratch/fschulz/tavd_g5_ce/model/best.pt \
  /netscratch/fschulz/tavd_g4_ce/model/best.pt \
  /netscratch/fschulz/tavd_g3_ce/model/best.pt \
  /netscratch/fschulz/tavd_g2_ce/model/best.pt \
  /netscratch/fschulz/tavd_xy_ce/best.pt \
  --data_root /netscratch/fschulz/test_landmarks_npy \
  --save_path /netscratch/fschulz/plot_rocs_g.png \
  --labels TAVD_G6 \
  TAVD_G5 \
  TAVD_G4 \
  TAVD_G3 \
  TAVD_G2 \
  TAVD_Baseline \
  --embedding_size 256
"""

# diddM and tavdM
"""
python plot_rocs2.py \
  --checkpoints /netscratch/fschulz/didd_xy_ce/best.pt \
  /netscratch/fschulz/tavd_xy_ce/best.pt \
  --data_root /netscratch/fschulz/didd_test_landmarks_npy \
  /netscratch/fschulz/test_landmarks_npy \
  --save_path /netscratch/fschulz/plot_rocs_didd_tavd.png \
  --labels DIDD_CE_XY \
    TAVD_CE_XY \
  --embedding_size 256
"""



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint .pt files or eval result dirs")
    p.add_argument("--data_root", nargs="+", required=True, help="One or more test .npy dataset roots. Provide one per checkpoint or a single root to use for all.")
    p.add_argument("--embedding_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_motion", action="store_true", help="Enable motion channels when building dataset")
    p.add_argument("--labels", nargs="+", default=None, help="Optional labels for legend (one per checkpoint)")
    p.add_argument("--save_path", type=str, default="rocs_compare.png")
    args = p.parse_args()

    plt.figure(figsize=(7,7))
    cmap = plt.get_cmap("tab10")

    total = len(args.checkpoints)
    for i, ck in enumerate(args.checkpoints):
        label = None
        # choose data_root for this checkpoint: per-checkpoint if provided, else use first
        if len(args.data_root) == len(args.checkpoints):
            data_root_for_ck = args.data_root[i]
        else:
            data_root_for_ck = args.data_root[0]

        # choose linestyle: dotted/dashdot for all except the last (which stays solid)
        if i == total - 1:
            ls = '-'            # last: solid
            lw = 2.0
        else:
            ls = ':' if (i % 2 == 0) else 'dashdot'  # alternate dotted / dashdot
            lw = 1.5

        # if ck is a directory with saved roc arrays, load them
        if os.path.isdir(ck):
            roc = get_roc_from_saved(ck)
            if roc is not None:
                fpr, tpr, roc_auc = roc
                label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(ck)
                plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})", color=cmap(i % 10), linestyle=ls, linewidth=lw)
                continue
            # else maybe user passed eval_dir but no arrays -> try to find eval_summary_compact.json for AUC only
            # fallthrough to recompute below using ck as checkpoint if file endswith .pt
        # if ck looks like a checkpoint file, compute embeddings+ROC
        if os.path.isfile(ck) and ck.endswith(".pt"):
            try:
                fpr, tpr, roc_auc = compute_embeddings_and_roc(ck, data_root_for_ck, args.embedding_size, args.device, args.use_motion)
            except Exception as e:
                print(f"[WARN] Failed computing ROC for {ck}: {e}")
                continue
            label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(ck)
            plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})", color=cmap(i % 10), linestyle=ls, linewidth=lw)
        else:
            print(f"[WARN] Unknown entry {ck}, skipping")

    plt.plot([0,1],[0,1],"k--", alpha=0.4)
    plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.save_path)
    print(f"[INFO] Saved combined ROC plot to {args.save_path}")

if __name__ == "__main__":
    main()