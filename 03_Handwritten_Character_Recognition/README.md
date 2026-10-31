# Task 3 — Handwritten Character Recognition

A convolutional neural network (CNN) trained on the MNIST handwritten digit dataset (0–9). TensorFlow/Keras automatically downloads MNIST on the first run.

## Run

```bash
pip install -r requirements.txt
python train.py --epochs 5
```

The script normalizes images, trains a CNN, evaluates test accuracy, saves `mnist_cnn.keras`, and writes a sample prediction image to `sample_predictions.png`.

## Extend to letters

Use the EMNIST dataset (for example, the Balanced or Letters split) and adjust the output layer and label mapping. EMNIST image orientation/transpose conventions should be checked before training. Full word/sentence recognition requires segmentation or a sequence model such as CRNN with CTC loss; it is not implemented in this digit baseline.
