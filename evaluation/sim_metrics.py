import numpy as np

# ---------------------------
# Similarity metrics
# ---------------------------

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def l2_distance(a, b):
    """Euklidische Distanz zwischen zwei Embeddings."""
    return float(np.linalg.norm(a - b))

def exp_similarity(a, b, tau=0.07):
    """
    Similarity nach Paper:
    exp(-||a-b||^2 / tau)

    tau ist ein Temperatur-Parameter (default hier 0.07,
    bitte anpassen an deinen Trainingswert).
    """
    dist_sq = np.sum((a - b) ** 2)
    return float(np.exp(-dist_sq / tau))