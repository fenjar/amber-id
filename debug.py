import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from glob import glob
import torch
from collections import defaultdict

from landmarks import extract_landmarks_mediapipe, compute_pairwise_normalized_distances, get_global_landmark_indices
import random

def aggregate_landmark_variances(video_paths, num_landmarks=126, norm="interocular", selected_indices=None):
    """
    Berechnet Varianz pro Landmark-Paar über mehrere Videos
    und erstellt ein Ranking der wichtigsten Paare.
    """

    # Akkumulator für Varianzen
    pair_variances = []

    for vp in video_paths:
        lm_list = extract_landmarks_mediapipe(vp, num_landmarks=num_landmarks, mode="2d", debug=True, selected_indices=selected_indices)
        if lm_list is None:
            print(f"[WARN] No landmarks extracted for video {vp}, skipping.")
            continue  # or return, depending on context
        lm_list = [lm for lm in lm_list if lm is not None]
        if len(lm_list) < 2:
            print(f"[WARN] {vp} zu kurz, übersprungen.")
            continue

        dist = compute_pairwise_normalized_distances(lm_list, norm=norm, selected_indices=selected_indices, debug=True)

        # If dist is a list of tuples (distance, i, j, region_i, region_j), extract only the numeric part
        if isinstance(dist, list) and isinstance(dist[0], (tuple, list)) and len(dist[0]) > 1:
            # dist: list of (distance, i, j, region_i, region_j)
            distances = np.array([[float(x[0]) for x in frame] for frame in dist])  # shape (T, D)
            pair_info = np.array([x for x in dist[0]])  # (D, 5)
        else:
            distances = np.array(dist)  # shape (T, D)
            pair_info = None

        var = distances.var(axis=0) # (D,)
        pair_variances.append(var)

        if pair_info is not None:
            all_pair_info = pair_info  # Save for later use outside the loop

    if not pair_variances:
        raise RuntimeError("Keine gültigen Videos gefunden!")

    # Mittelwert über alle Videos
    print(f"[INFO] Verarbeite {len(pair_variances)} Videos, jedes mit {pair_variances[0].shape[0]} Landmark-Paaren.")
    mean_var = np.mean(pair_variances, axis=0)  # (D,)

    # Ranking
    topk = np.argsort(mean_var)[::-1][:20]
    print("\n=== Top-20 variabelste Landmark-Paare für nvidia og ===")
    for rank, idx in enumerate(topk, 1):
        if pair_info is not None:
            i, j, region_i, region_j = int(all_pair_info[idx][1]), int(all_pair_info[idx][2]), str(all_pair_info[idx][3]), str(all_pair_info[idx][4])
            print(f"{rank:2d}. Paar-Index {idx:4d}  Varianz={mean_var[idx]:.6f}  (Landmarks {i}-{j}, Regionen {region_i}-{region_j})")
        else:
            print(f"{rank:2d}. Paar-Index {idx:4d}  Varianz={mean_var[idx]:.6f}")

    # Analyse nach Regionen
    if pair_info is not None:
        region_var = defaultdict(list)
        for idx in range(mean_var.shape[0]):
            region_pair = (str(all_pair_info[idx][3]), str(all_pair_info[idx][4]))
            region_var[region_pair].append(mean_var[idx])
        print("\n=== Durchschnittliche Varianz pro Regionenpaar ===")
        for region_pair, vars_ in sorted(region_var.items(), key=lambda x: -np.mean(x[1])):
            print(f"Regionen {region_pair[0]}-{region_pair[1]}: Mittelwert Varianz={np.mean(vars_):.6f} (Paare: {len(vars_)})")

    # Plot
    plt.figure(figsize=(12,5))
    plt.plot(mean_var, alpha=0.7)
    plt.title("Durchschnittliche Varianz pro Landmark-Paar über alle Videos")
    plt.xlabel("Landmark-Paar Index")
    plt.ylabel("Varianz")
    plt.tight_layout()
    plt.savefig("/netscratch/fschulz/avatar-fp-debug/aggregate_variance_plot-og.png")
    print("[INFO] Plot gespeichert unter aggregate_variance_plot-og.png")

    # Optional: Plot nach Regionen
    if pair_info is not None:
        plt.figure(figsize=(14,6))
        region_names = sorted(set([str(all_pair_info[idx][3]) for idx in range(mean_var.shape[0])] +
                    [str(all_pair_info[idx][4]) for idx in range(mean_var.shape[0])]))
        region_matrix = np.zeros((len(region_names), len(region_names)))
        count_matrix = np.zeros_like(region_matrix)
        region_idx_map = {name: i for i, name in enumerate(region_names)}
        for idx in range(mean_var.shape[0]):
            r1, r2 = str(all_pair_info[idx][3]), str(all_pair_info[idx][4])
            i, j = region_idx_map[r1], region_idx_map[r2]
            region_matrix[i, j] += mean_var[idx]
            count_matrix[i, j] += 1
        # Avoid division by zero
        region_matrix = np.divide(region_matrix, count_matrix, out=np.zeros_like(region_matrix), where=count_matrix!=0)
        im = plt.imshow(region_matrix, cmap="viridis")
        plt.colorbar(im, label="Durchschnittliche Varianz")
        plt.xticks(range(len(region_names)), region_names, rotation=90)
        plt.yticks(range(len(region_names)), region_names)
        plt.title("Regionenpaar-Varianz Heatmap")
        plt.tight_layout()
        plt.savefig("/netscratch/fschulz/avatar-fp-debug/aggregate_variance_region_heatmap-og.png")
        print("[INFO] Regionen-Heatmap gespeichert unter aggregate_variance_region_heatmap-og.png")

    return mean_var, topk


def debug_distance_variance(video_path, num_landmarks=80, clip_length=31, selected_indices=None):
    """
    Vergleicht die Varianz pro Landmark-Paar (über die Zeit)
    für BBox- und Interocular-Normalisierung.
    """
    # ---- Landmarks extrahieren ----
    lm_list = extract_landmarks_mediapipe(video_path, num_landmarks=num_landmarks, mode="2d", debug=True)
    lm_list = [lm for lm in lm_list if lm is not None]
    if len(lm_list) < clip_length:
        print(f"[WARN] Video {video_path} hat nur {len(lm_list)} Frames, zu kurz für {clip_length}.")
        return
    

    # ---- Distanzen für beide Normalisierungen ----
    dist_bbox = compute_pairwise_normalized_distances(lm_list, norm="bbox", selected_indices=selected_indices, debug=True)
    dist_inter = compute_pairwise_normalized_distances(lm_list, norm="interocular", selected_indices=selected_indices, debug=True)

    # Arrays formen (Frames × Distanzen)
    dist_bbox = np.array(dist_bbox)   # shape (T, D)
    dist_inter = np.array(dist_inter) # shape (T, D)

    # ---- Varianz pro Landmark-Paar über Zeit ----
    var_bbox = dist_bbox.var(axis=0)   # shape (D,)
    var_inter = dist_inter.var(axis=0) # shape (D,)

    # ---- Statistiken ausgeben ----
    print(f"[INFO] BBox-Varianz: mean={var_bbox.mean():.6f}, max={var_bbox.max():.6f}")
    print(f"[INFO] Interocular-Varianz: mean={var_inter.mean():.6f}, max={var_inter.max():.6f}")

    # ---- Plot Vergleich ----
    plt.figure(figsize=(12,5))
    plt.plot(var_bbox, label="BBox", alpha=0.7)
    plt.plot(var_inter, label="Interocular", alpha=0.7)
    plt.title("Varianz pro Landmark-Paar über Zeit")
    plt.xlabel("Landmark-Paar Index")
    plt.ylabel("Varianz")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/netscratch/fschulz/avatar-fp-debug/debug_distance_variance2.png")
    print("[INFO] Varianz-Plot gespeichert unter debug_distance_variance.png")

    # ---- Histogramme der Varianzen ----
    plt.figure(figsize=(12,5))
    plt.hist(var_bbox, bins=100, alpha=0.6, label="BBox", color="blue")
    plt.hist(var_inter, bins=100, alpha=0.6, label="Interocular", color="green")
    plt.title("Histogramm der Varianzwerte")
    plt.xlabel("Varianz")
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/netscratch/fschulz/avatar-fp-debug/debug_variance_hist2.png")
    print("[INFO] Histogramm gespeichert unter debug_variance_hist.png")

    return var_bbox, var_inter

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Videos und Labels laden
    # Erwartetes Format: data_root/idXXX/video.mp4
    selected_indices = get_global_landmark_indices(args.num_landmarks)

    if args.aggregate:
        # video_paths = [
        #     '/netscratch/fschulz/nvidia/nvidia_unzipped/id035/a01-id035.mp4',
        #     '/netscratch/fschulz/nvidia/nvidia_unzipped/id035/a03-id035.mp4',
        #     '/netscratch/fschulz/nvidia/nvidia_unzipped/id035/q03-id035.mp4',
        #     '/netscratch/fschulz/nvidia/nvidia_unzipped/id035/s05-id035.mp4',
        #     '/netscratch/fschulz/nvidia/nvidia_unzipped/id035/s07-id035.mp4',
        #     '/netscratch/fschulz/nvidia/nvidia_unzipped/id035/s14-id035.mp4',
        #     '/netscratch/fschulz/nvidia/nvidia_unzipped/id035/s22-id035.mp4',
        #     '/netscratch/fschulz/nvidia/nvidia_unzipped/id035/s28-id035.mp4',
        #     '/netscratch/fschulz/nvidia/nvidia_unzipped/id035/s30-id035.mp4',
        #     '/netscratch/fschulz/nvidia/nvidia_unzipped/id035/a07-id035.mp4'
        # ]
        video_paths = glob(os.path.join(os.path.dirname(args.video_path), "id*", "*.mp4"))

        # if len(video_paths) > 5:
        #     video_paths = random.sample(video_paths, 5)
        #     print(f"[INFO] Zufällige Auswahl von 5 Videos: {video_paths}")
        aggregate_landmark_variances(video_paths, num_landmarks=args.num_landmarks, selected_indices=selected_indices)
    else:
        video_path = str(args.video_path)
        debug_distance_variance(video_path, num_landmarks=args.num_landmarks, clip_length=args.clip_length, selected_indices=selected_indices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Pfad zum Video/Videos")
    parser.add_argument("--clip_length", type=int, default=31, help="Clip Length")
    parser.add_argument("--num_landmarks", type=int, default=80, help="Anzahl der Landmarks")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate over multiple videos")
    args = parser.parse_args()
    main(args)