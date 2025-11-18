# # video_deanon_train evaluation
Usage instructions for the evaluation scripts in the `evaluation` folder.

# Start model evaluation:

```bash

python eval.py \
  --checkpoint /path/to/model.pt \
  --data_root /netscratch/fschulz/datasets/avatars/test \
  --F 71 \
  --D 7875 \
  --num_samples 5

```

This will evaluate the model on the test set and print out the AUC.
For more options, run `python eval.py -h`.

# Train calibrator model:

```bash
python train_calibrator.py \
  --checkpoint /path/to/model.pt \
  --data_root /netscratch/fschulz/datasets/avatars/train \
  --F 71 \
  --D 7875 \
  --num_samples 5 \
  --out_dir /path/to/save/calibrator.pth

```

This will train a calibrator model on the training set and save it to the specified path. Also ROC curve will be plotted and saved in the same directory as PNG.
For more options, run `python train_calibrator.py -h`.

# Evaluate video pairs with/without trained calibrator:

```bash
python pair_predict.py \
    --checkpoint /path/to/model.pt \
    --videoA /path/to/id001_01.mp4 \
    --videoB /path/to/id002_01.mp4 \
    --F 71 \
    --D 7875 \
    --num_samples 5 \
    --calibration none|platt|isotonic \
    --calibrator /path/to/calibrator.joblib \
    --output_predictions /path/to/save/predictions.csv \

```
This will evaluate the specified video pairs and save the predictions to a CSV file.
For more options, run `python pair_predict.py -h`.

# Note:
- Ensure that the `--F` and `--D` parameters match those used during model training.
- The calibrator is optional; if not provided, the model will be evaluated without calibration.