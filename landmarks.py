import numpy as np
import cv2
import os
import mediapipe as mp
import random
import matplotlib.pyplot as plt

np.random.seed(6)  # Ensures reproducible landmark selection
random.seed(6)

def get_global_landmark_indices(num_landmarks, must_have=[133, 362], save_path="selected_indices.npy", exclude_regions=[]):
    """
    Selects and saves (or loads) a global set of landmark indices for all videos.
    """

    # contours = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_CONTOURS for item in sublist])))
    face_oval = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for item in sublist])))
    # irises = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_IRISES for item in sublist])))
    left_eye = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_LEFT_EYE for item in sublist])))
    right_eye = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE for item in sublist])))
    left_eyebrow = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW for item in sublist])))
    right_eyebrow = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW for item in sublist])))
    nose = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_NOSE for item in sublist])))
    lips = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_LIPS for item in sublist])))
    # tesselation = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_TESSELATION for item in sublist])))

    all = {
        "face": face_oval,
        "lips": lips,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "left_eyebrow": left_eyebrow,
        "right_eyebrow": right_eyebrow,
        "nose": nose
    }

    total_mp = 152 #contours + nose
    # total_mp = 468  # Standard MediaPipe FaceMesh
    selected_indices = []

    # --- Schritt 1: Alle möglichen Indizes bestimmen ---
    if exclude_regions is not None:
      for key in list(all.keys()):
        if key in exclude_regions:
          print(key)
          print(all[key])
          total_mp -= len(all[key])
          del all[key]
    # --- Schritt 2: Prozentuale Beteiligung der Indizes an num_landmarks ---
    nnn = 0
    if num_landmarks < total_mp:
        # iterate landmarks in remaining landmarks
        # total_mp updated when removed!! (schritt 1)
        for key, value in all.items():
          # formular: amount_of_lms*num_landmarks/total_mp
          amt = len(value) * num_landmarks / total_mp
          frac = amt - int(amt)
          nnn += frac
          if nnn >= 0.995:
            a = int(amt)+1
            nnn -= 1
          else:
            a = int(amt)

          selected_indices += (list(np.random.choice(value, a, replace=False)))
    else:
        print(f"[WARNING] num_landmarks {num_landmarks} >= total available {total_mp}, using all available.")
        selected_indices = list(range(0, total_mp))

    # --- Ensure 133 and 362 are always included for interocular normalization ---
    must_have = must_have if must_have is not None else []
    for idx in must_have:
        if idx not in selected_indices:
            # Remove a random index (not in must_have) to keep length fixed
            removable = [i for i in selected_indices if i not in must_have]
            if removable:
                selected_indices.remove(random.choice(removable))
            selected_indices.append(idx)
    
    # np.save(save_path, np.array(selected_indices))
    print(f"[INFO] Selected landmark indices ({len(selected_indices)}): {selected_indices}")

    return selected_indices

# ---------- Landmark-Exctraction with MediaPipe ----------
def extract_landmarks_mediapipe(video_path,
                               num_landmarks=126,
                               mode="2d",
                               debug=False,
                               selected_indices=None):

    face_detected = False  # Track if any face is detected

    mp_face = mp.solutions.face_mesh
    mp_detect = mp.solutions.face_detection

    cap = cv2.VideoCapture(video_path)
    lm_list = []

    if selected_indices is None:
        print("[INFO] No selected_indices provided, generating new selection.")
        selected_indices = get_global_landmark_indices(num_landmarks=num_landmarks)

    total_available = len(selected_indices)

    with mp_detect.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detect, mp_face.FaceMesh(static_image_mode=False,
                          max_num_faces=1,
                          refine_landmarks=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as face_mesh:

        frame_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break
            h, w = frame.shape[:2]

            # ---------- Schritt 1: Face Detection ----------
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_result = face_detect.process(rgb_frame)

            if detection_result.detections:
                face_detected = True  # At least one face detected
                # erste Face-Detection nehmen
                det = detection_result.detections[0]
                bboxC = det.location_data.relative_bounding_box
                x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Bounding Box sicher beschneiden
                pad = 0.2

                x1 = max(0, int(x - pad * bw))
                y1 = max(0, int(y - pad * bh))
                x2 = min(w, int(x + (1+pad) * bw))
                y2 = min(h, int(y + (1+pad) * bh))
                face_crop = frame[y1:y2, x1:x2]

                # ---------- Schritt 2: FaceMesh auf Bounding Box ----------
                frame_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(frame_rgb)

                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0]
                    if mode == "2d":
                        # pts = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
                        pts = np.array([[p.x * (x2 - x1), p.y * (y2 - y1)] for p in lm.landmark], dtype=np.float32)
                    elif mode == "3d":
                        # pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm.landmark], dtype=np.float32)
                        pts = np.array([[p.x * (x2 - x1), p.y * (y2 - y1), p.z * (x2 - x1)] for p in lm.landmark], dtype=np.float32)

                    # Wichtig: Landmarke wieder ins Koordinatensystem des Originals zurücklegen
                    pts[:, 0] += x1
                    pts[:, 1] += y1

                    pts_selected = pts[selected_indices]
                    lm_list.append(pts_selected)
                else:
                    lm_list.append(None)
            else:
                lm_list.append(None)

            frame_count += 1

    cap.release()
    # If no face was detected in any frame, skip this video
    if not face_detected:
        print(f"[WARNING] Skipping {video_path}: no face detected in any frame.")
        return None
    
    if debug:
        print(f"[DEBUG] Selected landmark indices ({len(selected_indices)}): {selected_indices}")
        print(f"[DEBUG] Total available landmarks after exclusion: {total_available}")
        return lm_list
    return lm_list

def compute_pairwise_normalized_distances(landmarks_list, norm="interocular", selected_indices=None, debug=False):

    # contours = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_CONTOURS for item in sublist])))
    face_oval = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for item in sublist])))
    # irises = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_IRISES for item in sublist])))
    left_eye = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_LEFT_EYE for item in sublist])))
    right_eye = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE for item in sublist])))
    left_eyebrow = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW for item in sublist])))
    right_eyebrow = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW for item in sublist])))
    nose = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_NOSE for item in sublist])))
    lips = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_LIPS for item in sublist])))
    # tesselation = np.array(list(set([item for sublist in mp.solutions.face_mesh.FACEMESH_TESSELATION for item in sublist])))

    all = {
        "face": face_oval,
        "lips": lips,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "left_eyebrow": left_eyebrow,
        "right_eyebrow": right_eyebrow,
        "nose": nose
    }

    landmark_pairs_with_regions = []

    first = next((x for x in landmarks_list if x is not None), None)
    if first is None:
        raise RuntimeError("No Landmarks found in any frame.")

    N = first.shape[0]
    idxs = [(i, j) for i in range(N) for j in range(i+1, N)]
    D = len(idxs)
    out = []

    # Find the mapping from original indices to selected_indices
    if selected_indices is not None:
        try:
            left_eye_inner_idx = selected_indices.index(133)
            right_eye_inner_idx = selected_indices.index(362)
        except ValueError:
            raise RuntimeError("Selected indices do not include 133 or 362!")
    else:
        left_eye_inner_idx = 133
        right_eye_inner_idx = 362

    # ---------- Video-weiter Normalisierungsfaktor ----------
    if norm == "bbox":
        widths, heights = [], []
        for lm in landmarks_list:
            if lm is not None:
                w = np.max(lm[:,0]) - np.min(lm[:,0])
                h = np.max(lm[:,1]) - np.min(lm[:,1])
                widths.append(w)
                heights.append(h)
        video_scale = np.mean([max(w,h) for w,h in zip(widths,heights)])
    elif norm == "interocular":
        scales = []
        for lm in landmarks_list:
            if lm is not None:
                scales.append(np.linalg.norm(lm[left_eye_inner_idx] - lm[right_eye_inner_idx]))
        video_scale = np.mean(scales)
    else:
        video_scale = 1.0

    if video_scale < 1e-6:
        video_scale = 1.0

    # ---------- Paarweise Distanzen ----------
    # Build a reverse lookup: landmark index -> region name
    idx_to_region = {}
    for region, indices in all.items():
        for idx in indices:
            idx_to_region[idx] = region

    for lm in landmarks_list:
        if lm is None:
            out.append(np.zeros(D, dtype=np.float32))
            landmark_pairs_with_regions.append([None] * D)
            continue
        dif = lm[:, None, :] - lm[None, :, :]
        dist = np.sqrt(np.sum(dif**2, axis=2))
        vec = dist[np.triu_indices(N, k=1)].astype(np.float32)
        vec = vec / video_scale

        out.append(vec)

        # For each pair, get the original landmark indices and their regions
        frame_pairs_info = []
        for idx, (i, j) in enumerate(idxs):
            # selected_indices maps from local index to original index
            orig_i = selected_indices[i] if selected_indices is not None else i
            orig_j = selected_indices[j] if selected_indices is not None else j
            region_i = idx_to_region.get(orig_i, "unknown")
            region_j = idx_to_region.get(orig_j, "unknown")
            frame_pairs_info.append((vec[idx], orig_i, orig_j, region_i, region_j))
        landmark_pairs_with_regions.append(frame_pairs_info)

    if debug:
        return landmark_pairs_with_regions

    return out

# # ---------- Pairwise Distances with Normalization ----------
# def compute_pairwise_normalized_distances(landmarks_list, norm="bbox"):
#     """
#     landmarks_list: Liste von (N,2) oder (N,3) Arrays (Frames x Landmarks).
#     norm: "bbox" (Breite/Höhe der Bounding Box) oder "interocular".
#     """
#     first = next((x for x in landmarks_list if x is not None), None)
#     if first is None:
#         raise RuntimeError("No Landmarks found in any frame.")

#     N = first.shape[0]
#     idxs = [(i, j) for i in range(N) for j in range(i+1, N)]
#     D = len(idxs)
#     out = []

#     # Definiere Landmark-Indices für Augen-Innenwinkel (Mediapipe FaceMesh Standard)
#     LEFT_EYE_INNER = 133
#     RIGHT_EYE_INNER = 362

#     for idx, lm in enumerate(landmarks_list):
#         if lm is None:
#             out.append(np.zeros(D, dtype=np.float32))
#             continue

#         # ----------- Normalisierungsfaktor bestimmen -----------
#         if norm == "bbox":
#             w = np.max(lm[:,0]) - np.min(lm[:,0])
#             h = np.max(lm[:,1]) - np.min(lm[:,1])
#             scale = max(w, h)  # oder np.sqrt(w**2 + h**2)
#         elif norm == "interocular":
#             scale = np.linalg.norm(lm[LEFT_EYE_INNER] - lm[RIGHT_EYE_INNER])
#         else:
#             scale = 1.0

#         if scale < 1e-6:
#             scale = 1.0

#         # ----------- Paarweise Distanzen berechnen -----------
#         dif = lm[:, None, :] - lm[None, :, :]
#         dist = np.sqrt(np.sum(dif**2, axis=2))
#         vec = dist[np.triu_indices(N, k=1)].astype(np.float32)

#         # # ----------- Normalisierung anwenden -----------
#         # vec = vec / scale

#         # Optional: z-Score Normalisierung innerhalb Frame
#         # mu, sigma = vec.mean(), vec.std()
#         # if sigma < 1e-6:
#         #     sigma = 1.0
#         # vec = (vec - mu) / sigma

#         out.append(vec)

#     return out

# ---------- Build Clips ----------
def build_sliding_clips(frame_vectors, clip_length=71, stride=1, flatten=True):
    F = clip_length
    T = len(frame_vectors)
    if T < F:
        return np.zeros((0, F * len(frame_vectors[0])), dtype=np.float32)
    D = frame_vectors[0].shape[0]
    num_clips = (T - F) // stride + 1
    clips = np.zeros((num_clips, F, D), dtype=np.float32)
    # print("Amount of Clips:", num_clips)
    for i in range(num_clips):
        for f in range(F):
            clips[i, f, :] = frame_vectors[i*stride + f]
    if flatten:
        clips = clips.reshape((num_clips, F * D))
    return clips

# ---------- Time-Shuffling (for R-Term) ----------
def time_shuffle_clip(clip_flat, clip_length, D):
    clip = clip_flat.reshape((clip_length, D))
    perm = np.random.permutation(clip_length)
    return clip[perm].reshape(-1)

# ---------- Pipeline ----------
def process_video_to_clips(video_path, num_landmarks=126, clip_length=75, stride=1, save_path=None, mode='2d', selected_indices=None):
    lm_list = extract_landmarks_mediapipe(video_path, num_landmarks=num_landmarks, mode=mode, selected_indices=selected_indices)
    if lm_list is None:
        print(f"[WARNING] Skipping {video_path}: no face detected in any frame.")
        return None
    valid_frames = [lm for lm in lm_list if lm is not None]

    if len(valid_frames) < clip_length:
        print(f"[WARNING] Skipping {video_path}: only {len(valid_frames)} valid frames (<{clip_length})")
        return None
    frame_vecs = compute_pairwise_normalized_distances(lm_list, norm="interocular", selected_indices=selected_indices)
    # print(f"[INFO] {len(frame_vecs)} Frames with {frame_vecs[0].shape[0]} distances each.")
    # print(f"[INFO] Example frame vector (first valid frame): {frame_vecs[next(i for i, v in enumerate(lm_list) if v is not None)]}")
    # print(f"[INFO] Example frame vector (second valid frame): {frame_vecs[next(i for i, v in enumerate(lm_list) if v is not None and i > next(j for j, vv in enumerate(lm_list) if vv is not None))]}")
    # print(f"[INFO] Example frame vector (last valid frame): {frame_vecs[next(i for i in reversed(range(len(lm_list))) if lm_list[i] is not None)]}")
    # print(f"[INFO] max std across all distances: {np.max(np.std([fv for fv in frame_vecs if fv is not None], axis=0))}")
    # return
    clips = build_sliding_clips(frame_vecs, clip_length=clip_length, stride=stride, flatten=False)
    if save_path:
        np.save(save_path, clips)
        print(f"[INFO] {clips.shape[0]} Clips saved at {save_path}")
    return clips