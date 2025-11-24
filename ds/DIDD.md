
# # DID-D Construction
Notes on reproducing, usage and licenses of the the foundational driving identity dataset (did-d).

# Source Datasets
DID-D is the collection of three source datasets: RAVDESS, CREMA-D, NVIDIA-VC Subset. The DID-D was created using the following ressources:
- [RAVDESS](https://zenodo.org/record/1188976)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- [NVFAIR-VS License available](https://research.nvidia.com/labs/nxp/nvfair/)

The system on which the DID-D is based follows the NVIDIA paper on Avatar Fingerprinting by Prashnani et al. (2024): [Avatar Fingerprinting](https://research.nvidia.com/labs/nxp/avatar-fingerprinting/)

To reconstruct the exact identity index order (RAVDESS (id001-id024), NVIDIA (id025-id070), CREMA-D (id071-id161)) from the Thesis, the following scripts were uses:

First, convert RAVDESS to the DID-D format:

```bash
#!/bin/bash
set -e

# ============================
# RAVDESS → did-d Structure
# ============================

SRC_BASE="/path/to/ravdess/ravdess_unzipped"
DST_BASE="/path/to/did-d"

for actor_dir in "$SRC_BASE"/Video_Speech_Actor_*; do
    [ -d "$actor_dir" ] || continue

    # Actor-ID extraction (e.g.. 09 from Video_Speech_Actor_09)
    raw_id=$(basename "$actor_dir" | grep -oE '[0-9]+$')
    actor_id=$(printf "%03d" "$((10#$raw_id))")  # force decimal

    src_videos="$actor_dir/Actor_${raw_id}"
    dst_dir="$DST_BASE/ravdess_id${actor_id}"
    mkdir -p "$dst_dir"

    echo ">> Copy videos from Actor ${actor_id} to ${dst_dir}"

    for video in "$src_videos"/*.mp4; do
        [ -f "$video" ] || continue
        cp "$video" "$dst_dir/"
    done
done

echo "✅ All RAVDESS-Videos copied to $DST_BASE."
```

Second, convert NVIDIA to the DID-D format:

```bash
#!/bin/bash
set -e

# ============================
# NVIDIA → did-d Stucture
# ============================

SRC_BASE="/path/to/nvidia/nvidia_unzipped"
DST_BASE="/path/to/did-d"
OFFSET=24  # RAVDESS got 24 IDs → NVIDIA starts at id025

for id_dir in "$SRC_BASE"/id0*; do
    [ -d "$id_dir" ] || continue

    # Extrahiere z.B. 014 aus id014
    raw_id=$(basename "$id_dir" | grep -oE '[0-9]+$')
    # force decimal
    dec_id=$((10#$raw_id))
    # Wende Offset an
    new_id=$((dec_id + OFFSET))
    # Formatting with leading zeros
    formatted_id=$(printf "%03d" "$new_id")

    dst_dir="$DST_BASE/nvidia_id${formatted_id}"
    mkdir -p "$dst_dir"

    echo ">> Copy Videos from ${id_dir} to ${dst_dir}"

    # Copy all .mp4 files (e.g. a01-id0xx.mp4, s03-id0xx.mp4, q03-id0xx.mp4, ...)
    for video in "$id_dir"/*.mp4; do
        [ -f "$video" ] || continue
        cp "$video" "$dst_dir/"
    done
done

echo "✅ All NVIDIA-Videos copied to $DST_BASE!"
```

Third and finally, convert CREMA-D to the DID-D format:
```bash
#!/bin/bash
set -e

# ============================
# CREMAD → did-d Structure
# ============================

SRC_BASE="/path/to/cremad/cremad_unzipped"
DST_BASE="/path/to/did-d"
OFFSET=70  # RAVDESS (24) + NVIDIA (46) = 70

mkdir -p "$DST_BASE"

# cover all mp4. files
for video in "$SRC_BASE"/*.mp4; do
    [ -f "$video" ] || continue

    # ID extraction: last two digits of leading nr
    filename=$(basename "$video")
    raw_prefix="${filename%%_*}"      # z.B. 1091
    raw_id="${raw_prefix: -2}"        # letzte zwei Ziffern → 91

    # leading zeros interpreting
    dec_id=$((10#$raw_id))
    new_id=$((dec_id + OFFSET))
    formatted_id=$(printf "%03d" "$new_id")

    dst_dir="$DST_BASE/cremad_id${formatted_id}"
    mkdir -p "$dst_dir"

    echo ">> Copy $filename → cremad_id${formatted_id}"
    cp "$video" "$dst_dir/"
done

echo "✅ All CREMAD-Videos copied to $DST_BASE!"
```

For the train/text/val split (k0), we used the same .sh as for the TAV-D, just modified to DID-D.
