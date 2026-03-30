# AMBER-ID: Avatar Motion-Based Embeddings for Re-Identification
Implementation of the Paper **Avatar Motion Signatures: Evaluating Linkability of Expressive De-Identification** [(Schulz et al., 2026)](https://www.scitepress.org/Link.aspx?doi=10.5220/0014622000004061) presented at the 12th International Conference on Information Systems Security and Privacy, awarded as [ICISSPs Best Paper 2026](https://icissp.scitevents.org/PreviousAwards.aspx) and published in the [Proceedings of the 12th International Conference on Information Systems Security and Privacy - (Volume 2)](https://www.scitepress.org/ProceedingsDetails.aspx?ID=/eschD8nZDc=&t=1).



AMBER-ID (Avatar Motion-Based Embedding for Re-Identification) is an end-to-end framework for evaluating the privacy of avatar-anonymized talking-head videos.
It investigates whether behavioral biometric information—specifically facial motion dynamics—can still be exploited to identify individuals, even when all appearance-based cues have been replaced by an avatar.

This repository contains the full implementation used in the thesis “Assessing Privacy of Anonymized Video”, including preprocessing, model training, and evaluation pipelines.

## 🔍 Overview
While avatarization removes physiological biometric information (appearance), it preserves facial motion to maintain expressiveness and usability.
However, motion patterns themselves can contain unique, identity-specific signatures.

AMBER-ID evaluates this privacy risk by training Spatio-Temporal Graph Convolutional Networks (ST-GCNs) on:

- Original talking-head videos → Baseline model

- Avatarized videos (via RAVAS) → Attacker model

and then comparing their verification performance.

## Key Features

✔️ ST-GCN-based motion representation learning

✔️ training pipeline adjustable to train a Baseline (refM) and Attacker (attM) models 

✔️ Complete privacy evaluation suite:

  - Equal Error Rate (EER)

  - ROC & AUC

  - Pairwise cosine similarity distributions

  - t-SNE visualization of the embedding space

✔️ Region-based ablation studies (eyes, lips, nose, eyebrows, face oval)

✔️ Applicable to any anonymization method or to clear data

✔️ Fully reproducible training & evaluation setup

## Method Summary

1️⃣ Datasets

The Driving Identity Dataset (DID-D) combines:

- RAVDESS — studio-quality emotional speech

- CREMA-D — diverse, crowd-sourced expressive speech

- NVIDIA-VS subset — scripted & free monologues in video-call conditions

All DID-D videos are processed through RAVAS to produce the Target Avatar Dataset (TAV-D).

2️⃣ Feature Extraction

Using MediaPipe FaceMesh:

- 2D facial landmarks (152 points)

- blendshape coefficients

- head pose matrices

are extracted and normalized to a canonical coordinate space.

3️⃣ Model — ST-GCN

The Spatio-Temporal Graph Convolutional Network:

- models spatial relations between landmarks

- captures temporal dynamics across frames

- is trained using cross-entropy classification

outputs embeddings suitable for 1:1 verification using cosine similarity

4️⃣ Evaluation Pipeline

Each model is evaluated on unseen identities using:

- ROC curves & AUC

- Equal Error Rate (EER)

- Accuracy at optimal threshold

- Cosine similarity distributions

- t-SNE embedding visualizations

📊 Results (Summary)

Motion alone is sufficient for identity inference → De-anonymization is possible. Avatarization reduces identity leakage → but does not eliminate it.

Best model performance:

- Baseline EER ≈ 0.16

- Attacker EER ≈ 0.26

Most identity-revealing motion regions: Eyes, Lips, Nose

## 🚀 Getting Started

To start Amber from scratch with your own video dataset, you just need to follow the steps described below.

### Foundational Dataset
You need a set of videos, organized in folders named by the logic `*_id*` where the first wildcard equals the identity source and the second wildcard equals the incremental numerical identifier, e.g. `ravdess_id001` or `cremad_id101`. The numerical id needs to be unique in the input dataset; one folder for each person. The source is technically irrelevant for the pipeline but handy for you to keep track of the source datasets. The naming of the videos inside each identity folder is technicaly irrelevant as well, but it is highly recommended to use a similar logic, this could be something like `ravdess_id001_video020`, for example.

`DATASET_ROOT/
  src1_id001/
    src1_id001_video001.mp4
    src1_id001_video002.mp4
    ...
  ...
  srcx_idyyy/
    srcx_idyyy_video001.mp4
    srcx_idyyy_video002.mp4
    ...`

The dataset will get split by the preprocessing pipeline. The dataset can be either an avatar dataset, an unanonymized dataset or a dataset with a complete different anonymization method, as long as the input videos contain human facial features. It is likely that strictly masked or permuted data will not result in meaningful embeddings, since Ambers landmarking system relies on spatial-temporal face landmarks. All kinds of human face synthesis or reenactments are the right choice for Amber. The preprocessing pipeline can detect one face per video.

Another important note: You need to perform a train-test-split beforehand. The model performs intrain validation on the train set, but you never want to evaluate the final model on seen data - so you need to split your set (80/20 or whatever you want) beforehand to have a hold-out test set for the final evaluation step. Then, perform the Preprocessing Pipeline on both the train and the test subset or perform the Prepocessing first and split the npy files into train and test afterwards and before training.

### Preprocessing Pipeline
The Preprocessing Pipeline will transfer raw input videos into landmark representations. The implementation can be found in [landmarks2.py](landmarks2.py). Define a output path for the npy and npz files for each video. it will be referred as `/path/to/outputs/landmarks_npy` and `/path/to/outputs/landmarks_meta`.

After you formatted your foundational dataset and selected/created the output paths, you can perform a quickstart. From your repo root (or wherever landmarks2.py is), run:

```
python landmarks2.py \
  --input_dir /path/to/DATASET_ROOT \
  --out_dir   /path/to/outputs/landmarks_npy \
  --meta_dir  /path/to/outputs/landmarks_meta \
  --t_max 71 \
  --groups all
```

What you’ll get:
* --out_dir/id001/<video_basename>.npy with shape (T_MAX, V, 4) where 4 = (x', y', dx, dy)
* --meta_dir/id001/<video_basename>.npz with orig_T, fps, bbox quantiles, and scale info
* Same for id002, etc.

It is important to mention that the preprocessing pipeline extracts not only 2D Landmarks over t_max frames per video, but it also extracts delta values for both coordinates (dx, dy). Those show the locational difference from the previous frame to the current one, it is a per-axis frame-to-frame difference. If you dont want to keep the deltas, you can just discard them.

If you set a higher `t_max value`, the videos will get sampled more detailled.

The `--groups all` flag is added to keep all landmark groups. It is also possible to perform ablation studies, so to mask one or more of the landmark sections. Avaulable landmark sections are face_oval, lips, left_eye, right_eye, left_eyebrow, right_eyebrow and nose. Check out the `GROUPS` enum of [landmarks2.py](landmarks2.py). `--groups face_oval, lips, nose` would mask the eye area, for example.

### STGCN Model Training
You can train directly from your .npy landmark clips with [stgcndual.py](stgcndual.py) as long as they’re laid out the way its dataset expects. The model loads data like this:

`/path/to/outputs/landmarks_npy/
  id_000/
    clip_000.npy
    clip_001.npy
  id_001/
    clip_000.npy
  ...`

Each .npy should be a NumPy array shaped:

* (T, V, 2) for x,y only, or
* (T, V, 4+) where the first 4 channels are x,y,dx,dy (it will slice what it needs)

If your files are (T, V, 2) and you don’t pass --no_motion, the script will compute dx,dy automatically. If you do pass --no_motion, it will use only (x,y).

The simplest way to start the pipeline is:

```
python stgcndual.py \
  --data_root /path/to/outputs/landmarks_npy \
  --save_dir /path/to/checkpoints/stgcn_nomotion_100e \
  --num_epochs 100 \
  --no_motion \
  --amp
```

This will train for 100 epochs, use positions only (x,y) due to --no_motion and use mixed precision on GPU (--amp) if available.

If you want a bit more control / stability:

```
python stgcndual.py \
  --data_root /path/to/outputs/landmarks_npy \
  --save_dir /path/to/checkpoints/stgcn_nomotion_100e \
  --num_epochs 100 \
  --warmup_epochs 10 \
  --L 64 \
  --batch_p 8 --batch_k 6 \
  --input_scale 10.0 \
  --no_motion \
  --num_workers 6 \
  --amp
```

* --L 64 means it will crop/pad every clip to 64 frames. Batch size is batch_p * batch_k (here 48).
* --input_scale scales x,y values; leave it at default unless you know your coordinate scale is already normalized differently.

### Evaluation

The evaluation pipeline can be found in [eval2.py](eval2.py). It is straight forward, you can simply run:

```
python eval2.py \
  --checkpoint /path/to/best.pt \
  --data_root /path/to/test_landmarks_npy \
  --save_dir /path/to/store/eval \
  --log_path /optional/path/to/logscript/slurm111.out \
  --match_root /path/to/test_landmarks_npy \
  --embedding_size 256
```
This will give you an evaluation in a 1:1 Verification manner (Same Pairs vs. Different Pairs).
