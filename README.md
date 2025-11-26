# AMBER-ID: Avatar Motion-Based Embeddings for Re-Identification
Implementation of the Master Thesis (TU Berlin, 2025)

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

To start Amber from scratch with your own video dataset, you just need to follow the steps described below. Prerequisite is a set of videos, organized in folders named by the logic `*_id*` where the first wildcard equals the source of the identity and the second wildcard equals the incremental numerical identifier, e.g. `ravdess_id001` or `cremad_id101`. The numerical id needs to be unique in the input dataset; one folder for each person. The source is irrelevant for the pipeline but handy to keep track of the source datasets. It will get split by the `landmarker.py` pipeline. The dataset can be either an avatar dataset, an unanonymized dataset or a dataset with a complete different anonymization method, as long as the input videos contain human facial features. It is likely that strictly masked or permuted data will not result in meaningful embeddings, since Ambers landmarking system relies on spatial-temporal face landmarks. All kinds of human face synthesis or reenactments are the right choice for Amber.

* Preprocessing and Landmarking
* STGCN Model Training
* Optional: Best Model Re-Selection on EER
* Evaluation on 1:1 Verification (Same Pairs vs. Different Pairs)
