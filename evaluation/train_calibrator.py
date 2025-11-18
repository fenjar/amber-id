import argparse
import os
import numpy as np
import torch
import joblib
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Import Processing function and model
from landmarks import process_video_to_clips
from model import TemporalIDNet3D
from evaluation.sim_metrics import cosine_similarity, l2_distance, exp_similarity

# ---------------------------
# Hilfsfunktionen
# ---------------------------

def load_model(checkpoint_path, device, F, D, embed_dim=128):
    model = TemporalIDNet3D(F=F, D=D, embed_dim=embed_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def embed_video(video_path, model, device, F, num_samples=5):
    clips = process_video_to_clips(video_path, clip_length=F, stride=1, num_landmarks=50, mode="2d")
    if len(clips) == 0:
        raise ValueError(f"No clips from {video_path}")

    if len(clips) >= num_samples:
        idxs = np.random.choice(len(clips), num_samples, replace=False)
    else:
        idxs = np.random.choice(len(clips), num_samples, replace=True)

    chosen = np.stack([clips[i] for i in idxs], axis=0)   # (k, F, D)
    x = torch.tensor(chosen, dtype=torch.float32, device=device)

    with torch.no_grad():
        z = model(x)  # (k,1,E) oder (k,5,E)
        if z.ndim == 3:
            if z.shape[1] == 1:
                z = z.squeeze(1)
            else:
                z = z.mean(dim=1)
        emb = z.cpu().numpy()

    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    agg = emb_norm.mean(axis=0)
    agg = agg / (np.linalg.norm(agg) + 1e-8)
    return agg

# ---------------------------
# Main
# ---------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Videos im Val-Set laden
    video_paths = glob(os.path.join(args.data_root, "*", "*.mp4"))
    labels = [os.path.basename(os.path.dirname(vp)) for vp in video_paths]
    print(f"Found {len(video_paths)} videos from {len(set(labels))} identities.")

    # Modell laden
    model = load_model(args.checkpoint, device, args.F, args.D)

    # Embeddings vorberechnen
    embeddings = []
    for vp in video_paths:
        emb = embed_video(vp, model, device, args.F, args.num_samples)
        embeddings.append(emb)
    embeddings = np.stack(embeddings, axis=0)

    # Similarities + Labels
    sims, y = [], []
    n = len(video_paths)
    for i in range(n):
        for j in range(i+1, n):
            if args.similarity_metric == "cosine":
                sim = cosine_similarity(embeddings[i], embeddings[j])
            elif args.similarity_metric == "l2":
                sim = l2_distance(embeddings[i], embeddings[j])
            elif args.similarity_metric == "exp":
                sim = exp_similarity(embeddings[i], embeddings[j])
            sims.append(sim)
            y.append(1 if labels[i] == labels[j] else 0)

    sims = np.array(sims)
    y = np.array(y)
    print(f"Generated {len(sims)} pairs for calibration.")

    # ---------------------------
    # Platt Scaling
    # ---------------------------
    platt = LogisticRegression(solver="lbfgs")
    platt.fit(sims.reshape(-1, 1), y)
    joblib.dump(platt, os.path.join(args.out_dir, "platt_calibrator.joblib"))
    print("Saved Platt calibrator to platt_calibrator.joblib")

    # ---------------------------
    # Isotonic Regression
    # ---------------------------
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(sims, y)
    joblib.dump(iso, os.path.join(args.out_dir, "isotonic_calibrator.joblib"))
    print("Saved Isotonic calibrator to isotonic_calibrator.joblib")

    # ---------------------------
    # Evaluation (AUC + ROC)
    # ---------------------------
    auc_raw = roc_auc_score(y, sims)
    auc_platt = roc_auc_score(y, platt.predict_proba(sims.reshape(-1, 1))[:, 1])
    auc_iso = roc_auc_score(y, iso.predict(sims))

    print("\n=== Validation AUC ===")
    print(f"Raw Cosine:   {auc_raw:.4f}")
    print(f"Platt Scale:  {auc_platt:.4f}")
    print(f"Isotonic:     {auc_iso:.4f}")

    # ROC Kurven berechnen
    fpr_raw, tpr_raw, _ = roc_curve(y, sims)
    fpr_platt, tpr_platt, _ = roc_curve(y, platt.predict_proba(sims.reshape(-1, 1))[:, 1])
    fpr_iso, tpr_iso, _ = roc_curve(y, iso.predict(sims))

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_raw, tpr_raw, label=f"Raw Cosine (AUC={auc_raw:.3f})")
    plt.plot(fpr_platt, tpr_platt, label=f"Platt (AUC={auc_platt:.3f})")
    plt.plot(fpr_iso, tpr_iso, label=f"Isotonic (AUC={auc_iso:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Calibration Methods")
    plt.legend(loc="lower right")
    plt.grid(True)

    out_path = os.path.join(args.out_dir, "roc_calibrators.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved ROC plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Pfad zu model.pt")
    parser.add_argument("--data_root", type=str, required=True, help="Pfad zum Validation-Set")
    parser.add_argument("--F", type=int, default=71, help="Clip Length")
    parser.add_argument("--D", type=int, default=7875, help="Feature Dimension")
    parser.add_argument("--num_samples", type=int, default=5, help="Clips pro Video")
    parser.add_argument("--out_dir", type=str, default=".", help="Output-Verzeichnis für Kalibratoren")
    parser.add_argument("--similarity_metric", type=str, default="cosine", choices=["cosine", "l2", "exp"], help="Ähnlichkeitsmetrik")
    args = parser.parse_args()
    main(args)
