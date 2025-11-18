import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from glob import glob

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


def embed_clips_from_video(video_path, model, device,
                           F, num_samples=5, deterministic=False):
    """
    Aggregiertes Embedding für ein Video erzeugen.
    - num_samples: wie viele Clips pro Video nutzen
    - deterministic: falls True → wähle z. B. den mittleren Clip
    """

    clips = process_video_to_clips(video_path, clip_length=F, stride=1, num_landmarks=80, mode="2d")  # (num_clips, F, D)
    if clips is None or len(clips) == 0:
        raise ValueError(f"No clips from video {video_path}")

    if deterministic:
        mid = len(clips) // 2
        idxs = [min(len(clips)-1, mid + i) for i in range(num_samples)]
    else:
        if len(clips) >= num_samples:
            idxs = random.sample(range(len(clips)), num_samples)
        else:
            idxs = [random.randrange(len(clips)) for _ in range(num_samples)]

    chosen = np.stack([clips[i] for i in idxs], axis=0)   # (k, F, D)
    x = torch.tensor(chosen, dtype=torch.float32, device=device)  # (k, F, D)

    with torch.no_grad():
        z = model(x)  # (k, 1, E) oder (k, 5, E)
        if z.ndim == 3:
            if z.shape[1] == 1:
                z = z.squeeze(1)  # (k, E)
            else:
                z = z.mean(dim=1)  # (k, E)
        emb = z.cpu().numpy()

    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    agg = emb_norm.mean(axis=0)  # (E,)
    agg = agg / (np.linalg.norm(agg) + 1e-8)
    return agg

# ---------------------------
# Evaluation
# ---------------------------

def evaluate_auc(video_paths, labels, model, device, F, num_samples=5):
    """
    video_paths: Liste von Videopfaden
    labels: Liste von Identity-Labels (z. B. id001, id002, ...)
    """
    embeddings = []
    for vp in video_paths:
        emb = embed_clips_from_video(vp, model, device, F, num_samples)
        embeddings.append(emb)
    embeddings = np.stack(embeddings, axis=0)

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

    auc = roc_auc_score(y, sims)
    return auc, sims, y


# ---------------------------
# Main
# ---------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Videos und Labels laden
    # Erwartetes Format: data_root/idXXX/video.mp4
    video_paths = glob(os.path.join(args.data_root, "*", "*.mp4"))
    labels = [os.path.basename(os.path.dirname(vp)) for vp in video_paths]

    print(f"Found {len(video_paths)} videos from {len(set(labels))} identities.")

    # load Model
    model = load_model(args.checkpoint, device, args.F, args.D)

    # calculate AUC
    auc, sims, y = evaluate_auc(video_paths, labels, model, device, args.F, num_samples=args.num_samples)
    print(f"Evaluation AUC: {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Pfad zu model.pt")
    parser.add_argument("--data_root", type=str, required=True, help="Pfad zum Test-Datenset")
    parser.add_argument("--F", type=int, default=71, help="Clip Length")
    parser.add_argument("--D", type=int, default=7875, help="Feature Dimension (z. B. 126 landmarks → D)")
    parser.add_argument("--num_samples", type=int, default=5, help="Wie viele Clips pro Video")
    parser.add_argument("--similarity_metric", type=str, default="cosine", choices=["cosine", "l2", "exp"], help="Ähnlichkeitsmetrik")
    args = parser.parse_args()
    main(args)
