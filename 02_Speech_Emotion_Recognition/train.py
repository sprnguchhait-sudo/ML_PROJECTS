"""MFCC-based speech emotion recognition baseline for RAVDESS-style filenames."""
import argparse
from pathlib import Path
import numpy as np
import librosa
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# RAVDESS emotion codes (third filename field).
EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}


def extract_features(path, duration=3.0, sr=22050):
    y, sr = librosa.load(path, sr=sr, duration=duration, offset=0.3)
    if y.size == 0:
        raise ValueError("Audio file contains no samples")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    # Mean and standard deviation preserve temporal summary while keeping a fixed-size vector.
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1),
                           delta.mean(axis=1), delta.std(axis=1)])


def load_dataset(root):
    X, y, groups = [], [], []
    files = sorted(Path(root).rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files found under {root}. See README for dataset setup.")
    for path in files:
        parts = path.stem.split("-")
        if len(parts) < 7 or parts[2] not in EMOTIONS:
            continue
        try:
            X.append(extract_features(path))
            y.append(EMOTIONS[parts[2]])
            groups.append(parts[6])  # actor ID
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
    if not X:
        raise ValueError("No valid RAVDESS-style files found; check filename convention.")
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Folder containing extracted WAV files")
    args = parser.parse_args()
    X, y, groups = load_dataset(args.data_dir)
    if len(np.unique(groups)) > 1:
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(split.split(X, y, groups))
    else:
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2,
                                               random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, probability=True,
                                                class_weight="balanced", random_state=42))
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    print("Accuracy:", round(accuracy_score(y[test_idx], pred), 4))
    print(classification_report(y[test_idx], pred, zero_division=0))
    print("Confusion matrix (alphabetical labels):\n",
          confusion_matrix(y[test_idx], pred, labels=sorted(set(y))))
    target = Path(__file__).resolve().parent / "speech_emotion_model.joblib"
    joblib.dump(model, target)
    print("Saved model:", target.name)


if __name__ == "__main__":
    main()
