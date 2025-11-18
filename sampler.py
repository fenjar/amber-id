import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
import os
from landmarks import process_video_to_clips

# ---------- AvatarDataset Class ----------

class AvatarClipDataset(Dataset):
    def __init__(self, driver_to_videos, clips_per_video=5, clips_per_id=16, clip_length=71, num_landmarks=64, mode="2d", preprocessed_dir=None, selected_indices=None):
        """
        driver_to_videos: dict {driver_id: list of video_data} --- IGNORE ---
        video_data: beliebige Struktur, aus der du Clips ziehen kannst --- IGNORE ---
        clips_per_video: wie viele Clips pro Video gesampelt werden --- IGNORE ---
        clip_length: Anzahl Frames pro Clip --- IGNORE ---
        num_landmarks: Anzahl der zu extrahierenden Landmarks --- IGNORE ---
        mode: '2d' oder '3d' für Landmark-Extraktion --- IGNORE ---
        """
        if preprocessed_dir is not None:
            self.is_preprocessed = True

        self.driver_to_videos = driver_to_videos
        self.driver_ids = list(driver_to_videos.keys())
        self.clips_per_video = clips_per_video
        self.clips_per_id = clips_per_id
        self.clip_length = clip_length + 4  # +4 für die 5er Entfaltung
        self.num_landmarks = num_landmarks
        self.mode = mode
        self.selected_indices = selected_indices

    def sample_clip(self, video):
        """
        video: path to video file or video data structure --- IGNORE ---
        Returns: Tensor of shape (F, D) --- IGNORE ---
        Dummy clip sampling
        - Extract frames
        - Compute landmark features
        - Return in (F, D) format
        """

        clips = process_video_to_clips( video,
                                        num_landmarks=self.num_landmarks,
                                        clip_length=self.clip_length, # is config clip_length + 4 for unfolding
                                        stride=1,
                                        mode=self.mode,
                                        selected_indices=self.selected_indices
                                    )

        # Skip videos where no clips could be extracted
        if clips is None:
            print(f"[SAMPLER] Warning: {video} is too short to be processed. Skipping this video.")
            return None

        if len(clips) == 0:
          # Return a tensor of zeros with the expected shape if no clips are extracted
          # This prevents errors but might indicate an issue with the video or processing
          print(f"[SAMPLER] Warning: No clips could be extracted from {video}. Returning zero tensor.")
          # Assuming D=7875 based on previous cell output, need to calculate D based on num_landmarks
          D = (self.num_landmarks * (self.num_landmarks - 1)) // 2 if self.mode == "2d" else (self.num_landmarks * (self.num_landmarks - 1)) // 2 # This needs to be calculated correctly based on mode and num_landmarks
          return torch.zeros((self.clip_length, D), dtype=torch.float32)


        # print("Clip-Array Shape:", clips.shape)
        # print("Preview of First Clip (clipped):", clips[0][:10])

        # choose a random clip
        idx = np.random.randint(0, len(clips))
        clip_np = clips[idx]  # shape (F+4, D)

        # transform to torch tensor
        clip_tensor = torch.tensor(clip_np, dtype=torch.float32)
        return clip_tensor  # (F+4, D)

    def __len__(self):
        print(f"[SAMPLER] Dataset Length: {len(self.driver_ids)}")
        return len(self.driver_ids)

    # Sampler for clips_per_video
    # def __getitem__(self, idx):
    #     driver_id = self.driver_ids[idx]
    #     videos = self.driver_to_videos[driver_id]
    #     clips = []
    #     meta = []
    #     for vid_idx, video in enumerate(videos):
    #         for _ in range(self.clips_per_video):
    #             clip_feat = self.sample_clip(video)
    #             clips.append(clip_feat)
    #             meta.append((driver_id, vid_idx, False, None))  # is_shuffled=False, no pair
    #     return clips, meta

    # Sampler for clips_per_id
    def __getitem__(self, idx):
        if hasattr(self, "is_preprocessed") and self.is_preprocessed:
            driver_id = self.driver_ids[idx]
            clip_files = self.driver_to_videos[driver_id]
            clips = []
            meta = []
            attempts = 0
            max_attempts = self.clips_per_id * 5  # Prevent infinite loop
            while len(clips) < self.clips_per_id and attempts < max_attempts:
                clip_file = random.choice(clip_files)
                try:
                    clip_np = np.load(clip_file)
                    clip_tensor = torch.tensor(clip_np, dtype=torch.float32)
                    clips.append(clip_tensor)
                    # Extract video index from filename if needed, else use -1
                    meta.append((driver_id, -1, False, None))
                except Exception as e:
                    print(f"[SAMPLER] Warning: Could not load {clip_file}: {e}")
                attempts += 1
            # If not enough valid clips, pad with zero tensors
            if len(clips) < self.clips_per_id:
                D = clips[0].shape[1] if clips else (self.num_landmarks * (self.num_landmarks - 1)) // 2
                pad_clip = torch.zeros((self.clip_length, D), dtype=torch.float32)
                for _ in range(self.clips_per_id - len(clips)):
                    clips.append(pad_clip)
                    meta.append((driver_id, -1, False, None))
            return clips, meta
        else:
            driver_id = self.driver_ids[idx]
            videos = self.driver_to_videos[driver_id]
            clips = []
            meta = []
            attempts = 0
            max_attempts = self.clips_per_id * 5  # Prevent infinite loop
            while len(clips) < self.clips_per_id and attempts < max_attempts:
                video = random.choice(videos)
                clip_feat = self.sample_clip(video)
                attempts += 1
                if clip_feat is None:
                    continue  # Skip if no clip could be sampled
                clips.append(clip_feat)
                meta.append((driver_id, videos.index(video), False, None))
            # If not enough valid clips, pad with zero tensors
            if len(clips) < self.clips_per_id:
                D = clips[0].shape[1] if clips else (self.num_landmarks * (self.num_landmarks - 1)) // 2
                pad_clip = torch.zeros((self.clip_length, D), dtype=torch.float32)
                for _ in range(self.clips_per_id - len(clips)):
                    clips.append(pad_clip)
                    meta.append((driver_id, -1, False, None))
            return clips, meta


# ---------- Helper Function for torch DataLoader ----------

def avatar_collate_fn(batch, shuffle_ratio=1.0):
    """
    Batch: list of (clips, meta)
    shuffle_ratio: proportion of clips for which we create a shuffled variant
    """
    all_clips = []
    driver_ids = []
    video_ids = []
    is_shuffled = []
    orig_pair_idx = []

    current_idx = 0
    for clips, meta in batch:
        for i, (driver_id, vid_id, _, _) in enumerate(meta):
            clip_feat = clips[i]
            all_clips.append(clip_feat)  # Append the full clip tensor
            driver_ids.append(driver_id)
            video_ids.append((driver_id, vid_id))
            is_shuffled.append(False)
            orig_pair_idx.append(-1)

            # evtl. Shuffle-Clip erzeugen
            if random.random() < shuffle_ratio:
                shuffled_clip = clip_feat[torch.randperm(clip_feat.size(0))]
                all_clips.append(shuffled_clip) # Append the full shuffled clip tensor
                driver_ids.append(driver_id)
                video_ids.append((driver_id, vid_id))
                is_shuffled.append(True)
                orig_pair_idx.append(current_idx)  # original idx
                orig_pair_idx[current_idx] = len(all_clips) - 1  # link back
            current_idx = len(all_clips)

    # convert to tensors
    # Stack the clips to form a tensor of shape (Total_clips, F, D)
    windows = torch.stack(all_clips, dim=0)
    unique_ids = {id_: idx for idx, id_ in enumerate(sorted(set(driver_ids)))}
    driver_ids_int = [unique_ids[id_] for id_ in driver_ids]
    driver_ids_tensor = torch.tensor(driver_ids_int, dtype=torch.long)
    #driver_ids_tensor = torch.tensor(driver_ids, dtype=torch.long)
    video_ids_tensor = torch.tensor(
        [hash(v) % (2**31-1) for v in video_ids], dtype=torch.long
    )
    is_shuffled_tensor = torch.tensor(is_shuffled, dtype=torch.bool)
    orig_pair_idx_tensor = torch.tensor(
        [idx if idx is not None else -1 for idx in orig_pair_idx], dtype=torch.long
    )

    return (
        windows, driver_ids_tensor, video_ids_tensor, is_shuffled_tensor, orig_pair_idx_tensor
        )

    # return {
    #     "windows": windows, # Return the clips under the key "windows"
    #     "embeddings": windows.mean(dim=1), # Also return the mean for compatibility with the loss function input expectation
    #     "driver_ids": driver_ids_tensor,
    #     "video_ids": video_ids_tensor,
    #     "clip_is_shuffled": is_shuffled_tensor,
    #     "original_pair_idx": orig_pair_idx_tensor,
    # }

# Helper functions for expanding metadata
def expand_by_5(tensor):
    """Expands a tensor by repeating each element 5 times."""
    return tensor.repeat_interleave(5)

def expand_pair_idx_by_5(tensor):
    """Expands pair indices, adjusting for the 5x expansion."""
    expanded_tensor = torch.zeros(tensor.size(0) * 5, dtype=tensor.dtype)
    for i in range(tensor.size(0)):
        if tensor[i] != -1:
            # Calculate the new index based on the 5x expansion
            expanded_tensor[i*5 : (i+1)*5] = tensor[i] * 5 + torch.arange(5)
        else:
             expanded_tensor[i*5 : (i+1)*5] = -1 # Keep -1 for no pair
    return expanded_tensor

# ---------- Build Dictionary for Driver_To_Videos ----------

# def build_driver_to_videos(root_dir):
#     driver_to_videos = {}
#     for driver_folder in sorted(os.listdir(root_dir)):
#         driver_path = os.path.join(root_dir, driver_folder)
#         # Skip files and directories that don't start with "id"
#         if not os.path.isdir(driver_path) or not driver_folder.startswith("id"):
#             continue
#         driver_id = int(driver_folder.replace("id", ""))  # z. B. "id001" -> 1
#         videos = []
#         for fname in sorted(os.listdir(driver_path)):
#             if fname.lower().endswith(".mp4"):
#                 videos.append(os.path.join(driver_path, fname))
#         driver_to_videos[driver_id] = videos
#     return driver_to_videos

# ---------- !IMPORTANT: Folder names for identites need to be unique ----------
def build_driver_to_videos(root_dir, preprocessed=False):
    driver_to_videos = {}
    for driver_folder in sorted(os.listdir(root_dir)):
        driver_path = os.path.join(root_dir, driver_folder)
        if not os.path.isdir(driver_path):
            continue
        # Use the folder name as the identity key (can be source_id0XX, etc.)
        driver_id = driver_folder
        videos = []
        for fname in sorted(os.listdir(driver_path)):
            if preprocessed:
                if fname.lower().endswith(".npy"):
                    videos.append(os.path.join(driver_path, fname))
            else:
                if fname.lower().endswith(".mp4"):
                    videos.append(os.path.join(driver_path, fname))
        if videos:  # Only add if there are videos
            driver_to_videos[driver_id] = videos
    return driver_to_videos


# ---------- PREPROCESS FOR preprocessing_only FLAG ----------
def preprocess_dataset(driver_to_videos, output_dir, clips_per_id, num_landmarks, clip_length, selected_indices):
    os.makedirs(output_dir, exist_ok=True)
    for driver_id, video_list in driver_to_videos.items():
        # Create a subfolder for each driver_id
        driver_dir = os.path.join(output_dir, driver_id)
        os.makedirs(driver_dir, exist_ok=True)
        for video_path in video_list:
            clips = process_video_to_clips(video_path, clip_length=clip_length, num_landmarks=num_landmarks, selected_indices=selected_indices)
            if clips is None or len(clips) == 0:
                print(f"[WARNING] Skipping {video_path}: not enough valid frames (<{clip_length})")
                continue  # Skip this video
            base_name = os.path.splitext(os.path.basename(video_path))[0]  # Remove .mp4 extension
            # for i, clip in enumerate(clips):
            #     out_path = os.path.join(driver_dir, f"{base_name}_clip{i}.npy")
            #     np.save(out_path, clip)

            # Randomly sample up to clips_per_id clips from all available clips
            num_clips = min(clips_per_id, len(clips))
            sampled_indices = np.random.choice(len(clips), num_clips, replace=False)
            for i, idx in enumerate(sampled_indices):
                clip = clips[idx]
                out_path = os.path.join(driver_dir, f"{base_name}_clip{i}.npy")
                np.save(out_path, clip)

# Example usage of the AvatarClipDataset and DataLoader
# dataset = AvatarClipDataset(driver_to_videos, clips_per_id=16)
# loader = DataLoader(dataset, batch_size=1, collate_fn=avatar_collate_fn)