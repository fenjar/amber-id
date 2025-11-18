import argparse
import numpy as np
import torch

# Import Processing function and model
from landmarks import process_video_to_clips
from model import TemporalIDNet3D
from evaluation.sim_metrics import cosine_similarity, l2_distance, exp_similarity


def load_model(checkpoint_path, device, F, D, embed_dim=128):
    model = TemporalIDNet3D(F=F, D=D, embed_dim=embed_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def embed_video(video_path, model, device, F, num_samples=5, deterministic=False):
    """
    Aggregiertes Embedding für ein einzelnes Video.
    """
    clips = process_video_to_clips(video_path, clip_length=F, stride=1, num_landmarks=50, mode="2d")
    if len(clips) == 0:
        raise ValueError(f"No clips extracted from {video_path}")

    if deterministic:
        mid = len(clips) // 2
        idxs = [min(len(clips)-1, mid + i) for i in range(num_samples)]
    else:
        if len(clips) >= num_samples:
            idxs = np.random.choice(len(clips), num_samples, replace=False)
        else:
            idxs = np.random.choice(len(clips), num_samples, replace=True)

    chosen = np.stack([clips[i] for i in idxs], axis=0)   # (k, F, D)
    x = torch.tensor(chosen, dtype=torch.float32, device=device)  # (k, F, D)

    with torch.no_grad():
        z = model(x)  # (k,1,E) oder (k,5,E)
        if z.ndim == 3:
            if z.shape[1] == 1:
                z = z.squeeze(1)  # (k, E)
            else:
                z = z.mean(dim=1)  # (k, E)
        emb = z.cpu().numpy()

    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    agg = emb_norm.mean(axis=0)
    agg = agg / (np.linalg.norm(agg) + 1e-8)
    return agg

# ---------------------------
# calibration methods
# ---------------------------

def heuristic_probability(sim, scale=5.0, bias=0.0):
    """Heuristische Sigmoid-Mapping."""
    return 1.0 / (1.0 + np.exp(-scale * (sim - bias)))


def platt_probability(sim, calibrator):
    """Platt Scaling (Logistic Regression)."""
    return calibrator.predict_proba(np.array([[sim]]))[:, 1][0]


def isotonic_probability(sim, calibrator):
    """Isotonic Regression."""
    return calibrator.predict(np.array([sim]))[0]


# ---------------------------
# Main
# ---------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args.F, args.D)

    embA = embed_video(args.videoA, model, device, args.F, args.num_samples)
    embB = embed_video(args.videoB, model, device, args.F, args.num_samples)

    if args.similarity_metric == "cosine":
        sim = cosine_similarity(embA, embB)
    elif args.similarity_metric == "l2":
        sim = l2_distance(embA, embB)
    elif args.similarity_metric == "exp":
        sim = exp_similarity(embA, embB)

    # Choose calibration type
    if args.calibration == "none":
        prob = heuristic_probability(sim)
    else:
        if args.calibrator is None:
            raise ValueError(f"--calibrator muss angegeben werden, wenn calibration={args.calibration}")
        calibrator = joblib.load(args.calibrator)
        if args.calibration == "platt":
            prob = platt_probability(sim, calibrator)
        elif args.calibration == "isotonic":
            prob = isotonic_probability(sim, calibrator)
        else:
            raise ValueError(f"Unbekannte calibration-Methode: {args.calibration}")

    print(f"Video A: {args.videoA}")
    print(f"Video B: {args.videoB}")
    print(f"{args.similarity_metric} Similarity: {sim:.4f}")
    print(f"Estimated Probability (same identity) [{args.calibration}]: {prob:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Pfad zu model.pt")
    parser.add_argument("--videoA", type=str, required=True, help="Erstes Video")
    parser.add_argument("--videoB", type=str, required=True, help="Zweites Video")
    parser.add_argument("--F", type=int, default=71, help="Clip Length")
    parser.add_argument("--D", type=int, default=7875, help="Feature Dimension")
    parser.add_argument("--num_samples", type=int, default=5, help="Clips pro Video")
    parser.add_argument("--calibration", type=str, choices=["none", "platt", "isotonic"], default="none",
                        help="Welche Kalibrationsmethode verwenden?")
    parser.add_argument("--calibrator", type=str, default=None,
                        help="Pfad zu gespeicherten Kalibrator (.joblib), wenn platt oder isotonic genutzt wird")
    parser.add_argument("--similarity_metric", type=str, default="cosine", choices=["cosine", "l2", "exp"], help="Ähnlichkeitsmetrik")
    args = parser.parse_args()
    main(args)
