"""CNN baseline for MNIST handwritten digit recognition."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
    x_test = x_test.astype("float32")[..., np.newaxis] / 255.0
    model = build_model()
    model.summary()
    model.fit(x_train, y_train, validation_split=0.1, epochs=args.epochs,
              batch_size=128, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}\nTest accuracy: {accuracy:.4f}")
    out = Path(__file__).resolve().parent
    model.save(out / "mnist_cnn.keras")
    probabilities = model.predict(x_test[:10], verbose=0)
    predictions = probabilities.argmax(axis=1)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i].squeeze(), cmap="gray")
        ax.set_title(f"Pred: {predictions[i]} | True: {y_test[i]}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "sample_predictions.png", dpi=150)
    plt.close(fig)
    print("Saved mnist_cnn.keras and sample_predictions.png")


if __name__ == "__main__":
    main()
