# # TAV-D Construction
Notes on reproducing the the target avatar dataset (tav-d).

# Sources and Ressources
TAV-D is based on DID-D and has the exact same identity indexing. Every video in did-d is mapped onto a single avatar with the help of the Anonymization Pipeline RAVAS:
- [RAVAS](https://github.com/carlosfranzreb/ravas)
- [Avatar Skin .glb File](https://github.com/carlosfranzreb/ravas/blob/main/rpm/public/avatar_1_f.glb)
- [Ready Player Me](https://readyplayer.me/)
- [Avatar LIVE DEMO](https://react-face-tracking.vercel.app)


Experiments were run on the AI High Performance Computing (HPC) Cluster of the [DFKI](https://www.dfki.de/web). To run the code headless, a virtual display was needed, created with xvfb.
A container environment with slurm got created based on the [Dockerfile](ravas/Dockerfile) from RAVAS `stream_processing`, `streamprocessing_xvfb.sqsh`.
The .yaml file to call the anonymization pipeline is equivalent to RAVAS [test_avatar.yaml](https://github.com/carlosfranzreb/ravas/blob/main/ravas/configs/test_avatar.yaml).

The following bash script was used to run the anonymizer and produce the TAV-D from DID-D on the RAVDESS identites. Same can be used to produce the avatarized version of the CREMA-D and NVIDIA-VS-SUBSET by adjusting the working directories and identity ranges.

RAVDESS identites from DID-D to TAV-D:

```bash
#!/bin/bash
#SBATCH --job-name=batch_avatar_ravdess
#SBATCH --output=slurm-%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --array=1-24

srun \
  --container-image=/path/to/image/streamprocessing_xvfb.sqsh \
  --container-workdir=/path/to/stream_processing \
  --container-mounts=/paths/to/mount:/paths/to/mount \
  bash -c "
    set -e

    cd /path/to/stream_processing

    echo '>>> Installing local package (stream_processing)'
    python setup.py install --user

    echo '>>> Starting batch anonymization'

    ACTOR_ID=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
    ORIGINALS=/path/to/didd/Video_Speech_Actor_${ACTOR_ID}
    AVATARS=/patch/to/tav-d/ravdess_avatars/id${ACTOR_ID}
    CONFIG_TEMPLATE=configs/test_avatar.yaml
    TMP_CONFIG=/tmp/tmp_avatar_${ACTOR_ID}.yaml

    # Output Directory erstellen
    mkdir -p \"\$AVATARS\"

    find \"\$ORIGINALS\" -type f -name \"*.mp4\" | while read -r video; do
        filename=\$(basename \"\$video\" .mp4)
        out_path=\"\$AVATARS/\$filename\"

        echo \"==> Processing \$filename\"

        # Dynamic config
        sed -E \"s|(define_video_file: \&video_file )[^ ]+|\1\$video|\" \"\$CONFIG_TEMPLATE\" \
        | sed \"s|^log_dir: .*|log_dir: \$out_path|\" > \"\$TMP_CONFIG\"

        # start to anonymize
        echo \"--- Processing \$video ---\"

        # starting Xvfb (virtual Display)
        Xvfb :88 -screen 0 1920x1080x24 &
        export DISPLAY=:88
        XVFB_PID=\$!

       # Starting Pipeline
       python -m stream_processing.main --config \"\$TMP_CONFIG\"

       # kill Xvfb
       kill \$XVFB_PID
       wait \$XVFB_PID 2>/dev/null

    done
  "
```

To create the test/train/val (k0) split used for training the attacker and baseline model, we used the following script:

```bash
#!/bin/bash
set -e

BASE="/path/to/tav-d"

# Create target folders
mkdir -p "$BASE/train" "$BASE/test" "$BASE/val"

# collect all ID-folders (e.g. ravdess_id001, nvidia_id050, cremad_id071, ...)
all_dirs=($(find "$BASE" -mindepth 1 -maxdepth 1 -type d -name "*_id*" -printf "%f\n"))

# optional: extract edge cases where anonymization failed (e.g. nvidia_id050)
special="nvidia_id050"
dirs=()
for d in "${all_dirs[@]}"; do
  if [[ "$d" != "$special" ]]; then
    dirs+=("$d")
  fi
done

# Shuffle all the other folders
shuffled=($(printf "%s\n" "${dirs[@]}" | shuf))

# Splits definition
num_test=35
num_val=14
num_train=112   # Rest

# optional: Test-Set gets the edge case
if (( RANDOM % 2 )); then
  test_set=("${shuffled[@]:0:$((num_test-1))}" "$special")
  val_set=("${shuffled[@]:$((num_test-1)):$num_val}")
  train_set=("${shuffled[@]:$((num_test-1+num_val))}")
else
  test_set=("${shuffled[@]:0:$num_test}")
  val_set=("$special" "${shuffled[@]:$num_test:$((num_val-1))}")
  train_set=("${shuffled[@]:$((num_test+num_val-1))}")
fi

# move
echo ">>> Copy Test-Set"
for d in "${test_set[@]}"; do
  mv "$BASE/$d" "$BASE/test/"
done

echo ">>> Copy Val-Set"
for d in "${val_set[@]}"; do
  mv "$BASE/$d" "$BASE/val/"
done

echo ">>> Copy Train-Set"
for d in "${train_set[@]}"; do
  mv "$BASE/$d" "$BASE/train/"
done

echo "Done! Split:"
echo "Test: ${#test_set[@]} Folders"
echo "Val: ${#val_set[@]} Folders"
echo "Train: ${#train_set[@]} Folders"
```
