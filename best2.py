"""
Evaluate all .pt checkpoints in a directory on a held-out test set and pick the checkpoint
with the lowest EER. The selected checkpoint is copied to <ckpt_dir>/eer_best.pt.

Usage example:
python best2.py \
  --ckpt_dir /netscratch/fschulz/tavd_g1_ce \
  --data_root /netscratch/fschulz/g1_test_landmarks_npy \
  --device cuda \
  --embedding_size 256 \
  --match_root /netscratch/fschulz/g1_test_landmarks_npy \
  --save_json
"""
import os
import argparse
import json
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

from stgcndual import STGCNBackbone, LandmarkVideoDataset

def compute_eer_from_embeddings(embeddings, labels):
    """
    embeddings: torch.Tensor (N, D) on CPU or numpy
    labels: 1D array-like of length N (ints)
    returns: (eer, roc_auc) floats; eer = np.inf if cannot compute
    """
    if isinstance(embeddings, torch.Tensor):
        embs = embeddings.cpu().numpy()
    else:
        embs = np.asarray(embeddings)
    labels = np.asarray(labels)
    N = embs.shape[0]
    if N < 2:
        return float("inf"), float("nan")
    sim_mat = embs @ embs.T
    iu = np.triu_indices(N, k=1)
    sims = sim_mat[iu]
    targets = (labels[iu[0]] == labels[iu[1]]).astype(int)
    if targets.sum() == 0 or (targets == 0).sum() == 0:
        return float("inf"), float("nan")
    fpr, tpr, _ = roc_curve(targets, sims)
    roc_auc = float(auc(fpr, tpr))
    eer = float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])
    return eer, roc_auc

def load_embeddings(model, dataset, device):
    model.eval()
    embs = []
    labs = []
    with torch.no_grad():
        for x, label in dataset:
            x = x.unsqueeze(0).to(device)
            z = model(x)
            z = F.normalize(z, p=2, dim=1)
            embs.append(z.cpu())
            labs.append(int(label))
    if len(embs) == 0:
        return None, None
    embs = torch.cat(embs, dim=0)
    return embs, np.array(labs, dtype=int)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True, help="Directory with .pt checkpoints")
    p.add_argument("--data_root", type=str, required=True, help="Path to test .npy dataset (same format as LandmarkVideoDataset)")
    p.add_argument("--embedding_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--match_root", type=str, default=None, help="Optional: restrict test set to files also present in this root")
    p.add_argument("--out_name", type=str, default="eer_best.pt", help="Filename to write the best checkpoint to inside ckpt_dir")
    p.add_argument("--save_json", action="store_true", help="Save per-checkpoint metrics into ckpt_dir/ckpt_eer_results.json")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt_files = sorted([os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir) if f.endswith(".pt")])
    if not ckpt_files:
        raise SystemExit(f"No .pt files found in {args.ckpt_dir}")

    # build test dataset (deterministic)
    test_dataset = LandmarkVideoDataset(
        data_root=args.data_root,
        L=64,
        augmentation=False,
        input_scale=10.0,
        motion_scale=20.0,
        seed=42
    )
    if args.match_root:
        match_basenames = {f for _, _, files in os.walk(args.match_root) for f in files if f.endswith(".npy")}
        orig = len(test_dataset.samples)
        test_dataset.samples = [(lbl, fp) for (lbl, fp) in test_dataset.samples if os.path.basename(fp) in match_basenames]
        kept = len(test_dataset.samples)
        print(f"[INFO] match_root provided -> kept {kept}/{orig} samples")
        if kept == 0:
            raise SystemExit("No overlapping .npy files between data_root and match_root")

    results = []
    for ckpt in ckpt_files:
        try:
            ck = torch.load(ckpt, map_location=device)
        except Exception as e:
            print(f"[WARN] Could not load {ckpt}: {e}")
            continue

        # instantiate model with requested embedding size and dataset V/in_channels
        # need a sample to determine V and in_channels
        try:
            sample_x, _ = test_dataset[0]
        except Exception as e:
            raise SystemExit(f"Failed to read a sample from test dataset: {e}")
        in_channels = sample_x.shape[0]
        V = sample_x.shape[2]

        model = STGCNBackbone(in_channels=in_channels, num_layers=9, V=V, base_channels=64, embedding_size=args.embedding_size, dropout=0.3).to(device)
        # load backbone weights (non-strict to allow small shape differences)
        try:
            model.load_state_dict(ck.get("backbone", ck), strict=False)
        except Exception as e:
            # fallback: try loading entire checkpoint if it contains model directly
            try:
                model.load_state_dict(ck, strict=False)
            except Exception:
                print(f"[WARN] Could not load backbone weights from {ckpt}: {e}")
                continue

        # compute embeddings + labels
        embs, labels = load_embeddings(model, test_dataset, device)
        if embs is None:
            print(f"[WARN] No embeddings produced for {ckpt}; skipping")
            continue

        eer, roc_auc = compute_eer_from_embeddings(embs, labels)
        print(f"[INFO] {os.path.basename(ckpt):30s}  EER={eer:.4f}  AUC={roc_auc:.4f}")
        results.append({"ckpt": ckpt, "eer": eer, "auc": roc_auc})

    if not results:
        raise SystemExit("No valid checkpoint evaluations produced any metrics.")

    # sort by EER ascending (lower better), break ties by higher AUC
    results.sort(key=lambda r: (r["eer"], - (r["auc"] if r["auc"] is not None else -1.0)))
    best = results[0]
    best_ckpt = best["ckpt"]
    print(f"[RESULT] Best by EER: {os.path.basename(best_ckpt)}  EER={best['eer']:.4f}  AUC={best['auc']:.4f}")

    # copy best checkpoint to ckpt_dir/out_name
    dest = os.path.join(args.ckpt_dir, args.out_name)
    shutil.copy(best_ckpt, dest)
    print(f"[INFO] Copied best checkpoint to {dest}")

    if args.save_json:
        json_path = os.path.join(args.ckpt_dir, "ckpt_eer_results.json")
        with open(json_path, "w") as jf:
            json.dump(results, jf, indent=2)
        print(f"[INFO] Saved per-checkpoint results to {json_path}")

if __name__ == "__main__":
    main()