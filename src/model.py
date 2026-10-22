"""Phase 1, step 1: train a small float MLP (64 -> 32 -> 10, ReLU) on sklearn digits.

Everything is seeded (SEED = 0) so the numbers are reproducible.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

SEED = 0


def load_data():
    """Return X_train, X_test, y_train, y_test. Pixels are scaled from 0..16 to 0..1."""
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def train_float(X_train, y_train):
    """Train the MLP and return its parameters as a dict of NumPy arrays.

    Weight convention used everywhere in this project: y = x @ W + b,
    so W1 has shape (64, 32) and W2 has shape (32, 10).
    """
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                        solver="adam", max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    return {
        "W1": clf.coefs_[0].copy(), "b1": clf.intercepts_[0].copy(),
        "W2": clf.coefs_[1].copy(), "b2": clf.intercepts_[1].copy(),
    }


def float_forward(X, p):
    """Float forward pass. Returns (logits, hidden activations)."""
    h = np.maximum(X @ p["W1"] + p["b1"], 0.0)
    return h @ p["W2"] + p["b2"], h


def accuracy(logits, y):
    return float(np.mean(np.argmax(logits, axis=1) == y))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    p = train_float(X_train, y_train)
    print(f"train size {len(y_train)}, test size {len(y_test)}")
    print(f"float test accuracy: {accuracy(float_forward(X_test, p)[0], y_test):.4f}")
