"""
extract_landmarks_normalized.py

Produces:
 - data:  .npy  file per input video       -> shape (T_MAX, V, 4)  (x', y', dx, dy)
 - meta:  .npz  file per input video       -> contains orig_T, fps, bbox_quantiles, scale_used

Requirements:
 - mediapipe
 - opencv-python
 - numpy
 - scipy (for resample)
"""

import os
import json
import numpy as np
import cv2
from scipy.signal import resample
import mediapipe as mp
from glob import glob
import argparse

# -------------- defaults / settings --------------
T_MAX = 71
OUT_DIR = "/netscratch/fschulz/g1_test_landmarks_npy"
META_DIR = "/netscratch/fschulz/g1_test_landmarks_meta"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# build named landmark groups (unique indices per group)
def _unique_indices_from(facemesh_attr):
    return np.array(list(set([item for sublist in facemesh_attr for item in sublist])))

GROUPS = {
    "face_oval": _unique_indices_from(mp.solutions.face_mesh.FACEMESH_FACE_OVAL),
    "lips": _unique_indices_from(mp.solutions.face_mesh.FACEMESH_LIPS),
    "left_eye": _unique_indices_from(mp.solutions.face_mesh.FACEMESH_LEFT_EYE),
    "right_eye": _unique_indices_from(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE),
    "left_eyebrow": _unique_indices_from(mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW),
    "right_eyebrow": _unique_indices_from(mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW),
    "nose": _unique_indices_from(mp.solutions.face_mesh.FACEMESH_NOSE),
}

# default selection = all groups (keeps previous behavior)
all_landmark_indices = np.unique(np.concatenate(list(GROUPS.values())))
N_LANDMARKS_SELECTED = int(len(all_landmark_indices))

def set_landmark_selection(names):
    """
    names: comma-separated string or list of group keys from GROUPS or "all".
    Updates global all_landmark_indices and N_LANDMARKS_SELECTED.
    """
    global all_landmark_indices, N_LANDMARKS_SELECTED
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip() != ""]
    if len(names) == 0 or (len(names) == 1 and names[0].lower() == "all"):
        all_landmark_indices = np.unique(np.concatenate(list(GROUPS.values())))
    else:
        picked = []
        for n in names:
            if n not in GROUPS:
                raise ValueError(f"Unknown landmark group '{n}'. Valid: {sorted(GROUPS.keys())}")
            picked.append(GROUPS[n])
        all_landmark_indices = np.unique(np.concatenate(picked))
    N_LANDMARKS_SELECTED = int(len(all_landmark_indices))

# mediapipe init
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# helper: linear interpolation for missing frames
def interp_missing(frames):
    # frames: (T, V, 2) with NaNs where missing
    T = frames.shape[0]
    for v in range(frames.shape[1]):
        for dim in range(frames.shape[2]):
            arr = frames[:, v, dim]
            mask = np.isfinite(arr)
            if mask.sum() == 0:
                # all missing -> zeros
                frames[:, v, dim] = 0.0
            elif mask.sum() == 1:
                # fill with the single observed value
                frames[:, v, dim] = arr[mask][0]
            else:
                idx = np.arange(T)
                frames[:, v, dim] = np.interp(idx, idx[mask], arr[mask])
    return frames

def extract_and_normalize(video_path, t_max=T_MAX, quantile_low=0.02, quantile_high=0.98, target_scale=1.0, eps=1e-6):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else None
    frames_coords = []  # list of (V,2) for selected landmarks
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            # Select only the desired landmarks
            coords = np.array([[lm.landmark[i].x, lm.landmark[i].y] for i in all_landmark_indices], dtype=np.float32)  # (V=N_LANDMARKS_SELECTED, 2), normalized 0..1
        else:
            # mark missing frames as NaN to allow interpolation later
            coords = np.full((N_LANDMARKS_SELECTED, 2), np.nan, dtype=np.float32)
        frames_coords.append(coords)

    cap.release()
    if len(frames_coords) == 0:
        return None, None

    data = np.stack(frames_coords, axis=0)  # (T, V=N_LANDMARKS_SELECTED, 2)
    orig_T = data.shape[0]

    # interpolate missing frames (per landmark)
    if np.isnan(data).any():
        data = interp_missing(data)

    # compute robust global bbox quantiles over all frames & landmarks
    xs = data[..., 0].reshape(-1)
    ys = data[..., 1].reshape(-1)
    # use only finite (should be after interp)
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    qx0 = np.quantile(xs, quantile_low)
    qx1 = np.quantile(xs, quantile_high)
    qy0 = np.quantile(ys, quantile_low)
    qy1 = np.quantile(ys, quantile_high)
    bbox = (qx0, qy0, qx1, qy1)  # xmin, ymin, xmax, ymax

    # center and scale (global)
    cx = 0.5 * (qx0 + qx1)
    cy = 0.5 * (qy0 + qy1)
    width = (qx1 - qx0)
    height = (qy1 - qy0)
    scale = max(width, height)
    if scale < eps:
        scale = eps

    # normalize so that scale -> target_scale (commonly 1.0)
    data_norm = (data - np.array([cx, cy])[None, None, :]) / scale * target_scale
    # data_norm shape (T, V=N_LANDMARKS_SELECTED, 2)

    # compute deltas (dx, dy) in normalized coordinates
    d = np.diff(data_norm, axis=0, prepend=data_norm[0:1])
    combined = np.concatenate([data_norm, d], axis=-1)  # (T, V=N_LANDMARKS_SELECTED, 4)

    # resample temporally to t_max
    if combined.shape[0] != t_max:
        combined = resample(combined, t_max, axis=0)

    # pack metadata
    meta = {
        "orig_T": int(orig_T),
        "fps": float(fps) if fps is not None else None,
        "bbox_quantiles": [float(qx0), float(qy0), float(qx1), float(qy1)],
        "center": [float(cx), float(cy)],
        "scale_used": float(scale),
        "target_scale": float(target_scale)
    }

    return combined.astype(np.float32), meta

# batch processing
def process_folder(input_dir, output_dir=OUT_DIR, meta_dir=META_DIR, t_max=T_MAX):
    # Iterate through subdirectories matching the pattern
    for subdir in glob(os.path.join(input_dir, '*_id*')):
        if not os.path.isdir(subdir):
            continue

        subdir_name = os.path.basename(subdir)
        output_subdir_data = os.path.join(output_dir, subdir_name)
        output_subdir_meta = os.path.join(meta_dir, subdir_name)
        os.makedirs(output_subdir_data, exist_ok=True)
        os.makedirs(output_subdir_meta, exist_ok=True)

        files = sorted([f for f in os.listdir(subdir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
        for fn in files:
            inpath = os.path.join(subdir, fn)
            base = os.path.splitext(fn)[0]
            out_np = os.path.join(output_subdir_data, base + '.npy')
            out_meta = os.path.join(output_subdir_meta, base + '.npz')

            if os.path.exists(out_np) and os.path.exists(out_meta):
                print(f"[SKIP] {os.path.join(subdir_name, base)}")
                continue

            # print(f"[INFO] Processing {os.path.join(subdir_name, fn)} ...")
            data, meta = extract_and_normalize(inpath, t_max=t_max)
            if data is None:
                print(f"[WARN] no frames for {os.path.join(subdir_name, fn)}")
                continue
            np.save(out_np, data)                    # shape (T_MAX, V, 4)
            np.savez_compressed(out_meta, **meta)    # store metadata
            print(f"[OK] saved {os.path.join(subdir_name, base)}.npy  meta->{os.path.join(subdir_name, base)}.npz  shape={data.shape}")


def build_argparser():
    p = argparse.ArgumentParser(description="Extract & normalize landmarks (landmarks2.py)")
    p.add_argument("--input_dir", type=str, required=True, help="Input folder containing *_id* subfolders with videos")
    p.add_argument("--out_dir", type=str, default=OUT_DIR, help="Output directory for .npy files")
    p.add_argument("--meta_dir", type=str, default=META_DIR, help="Output directory for .npz metadata")
    p.add_argument("--t_max", type=int, default=T_MAX, help="Number of frames to resample to (T_MAX)")
    p.add_argument("--groups", type=str, default="all",
                   help="Comma-separated landmark groups to keep (available: {}) or 'all'".format(", ".join(sorted(GROUPS.keys()))))
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    # apply group selection
    set_landmark_selection(args.groups)
    # update dirs / t_max
    OUT_DIR = args.out_dir
    META_DIR = args.meta_dir
    T_MAX = args.t_max
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    print(f"[INFO] Using landmark groups: {args.groups} -> {N_LANDMARKS_SELECTED} landmarks")
    process_folder(args.input_dir, output_dir=OUT_DIR, meta_dir=META_DIR, t_max=T_MAX)