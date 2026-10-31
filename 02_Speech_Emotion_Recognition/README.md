# Task 2 — Emotion Recognition from Speech

This starter extracts MFCC, delta-MFCC, and summary audio features with librosa, then trains an SVM classifier. It demonstrates the signal-processing and classification workflow without requiring a GPU.

## Dataset setup

Download the **RAVDESS Audio-Only** dataset from its official Zenodo record: https://zenodo.org/records/1188976 . Read and comply with the dataset terms. Extract the audio files into `data/`. This project expects filenames containing the RAVDESS emotion code in the third hyphen-separated field (for example, `03-01-05-...wav`). It uses these categories: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

## Run

```bash
pip install -r requirements.txt
python train.py --data-dir data
```

The script reports held-out classification metrics and saves `speech_emotion_model.joblib`. The default script uses speaker-independent grouping by actor for its test split when enough actors are available.

Emotion labels describe acted recordings in a dataset. They are not reliable proof of a speaker's actual feelings.
