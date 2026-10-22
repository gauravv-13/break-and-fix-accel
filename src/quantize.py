"""Phase 1, step 2: symmetric per-layer INT8 quantization.

How it works (simple version):
- Each weight matrix gets one scale s_w = max|W| / 127, and W_q = round(W / s_w) in [-127, 127].
- Each activation tensor (the input and the hidden layer) gets one scale s_x = max|x| / 127,
  measured on the training set ("calibration").
- Matrix multiplies are done on INT8 numbers with 32-bit integer accumulators.
- Biases are stored as int32 in the accumulator's scale: b_q = round(b / (s_x * s_w)).
- After layer 1 we apply ReLU and "requantize" the int32 result back to INT8 with
  the multiplier m1 = s_in * s_w1 / s_h. (We apply m1 as a float multiply; real hardware
  would use a fixed-point multiplier and shift. This is a known simplification.)
- Layer 2's int32 result is the output; argmax of it gives the predicted digit.
"""
from pathlib import Path

import numpy as np

from src import model as model_mod

QMAX = 127
RESULTS = Path(__file__).resolve().parent.parent / "results" / "phase1"
NPZ_PATH = RESULTS / "model_int8.npz"


def symmetric_scale(x):
    return float(np.max(np.abs(x))) / QMAX


def quantize_tensor(x, scale):
    return np.clip(np.round(x / scale), -QMAX, QMAX).astype(np.int8)


def quantize_input(X, s_in):
    """Float inputs -> INT8 inputs."""
    return quantize_tensor(X, s_in)


def quantize_model(p, X_calib):
    """Quantize float params p using X_calib (training data) to pick activation scales."""
    _, h = model_mod.float_forward(X_calib, p)
    s_in = symmetric_scale(X_calib)
    s_h = symmetric_scale(h)
    s_w1 = symmetric_scale(p["W1"])
    s_w2 = symmetric_scale(p["W2"])
    return {
        "W1q": quantize_tensor(p["W1"], s_w1),
        "W2q": quantize_tensor(p["W2"], s_w2),
        "b1q": np.round(p["b1"] / (s_in * s_w1)).astype(np.int32),
        "b2q": np.round(p["b2"] / (s_h * s_w2)).astype(np.int32),
        "s_in": s_in, "s_h": s_h, "s_w1": s_w1, "s_w2": s_w2,
        "m1": s_in * s_w1 / s_h,          # requantization multiplier after layer 1
        "out_scale": s_h * s_w2,          # int32 logits * out_scale = float logits
    }


def int_matmul(Xq, Wq):
    """Reference INT8 x INT8 -> INT32 matrix multiply (32-bit accumulators)."""
    return Xq.astype(np.int32) @ Wq.astype(np.int32)


def postprocess_hidden(acc1, q):
    """Bias add, ReLU and requantize layer-1 int32 results to INT8 (range 0..127)."""
    z = acc1 + q["b1q"]
    z = np.maximum(z, 0)
    return np.clip(np.round(z * q["m1"]), 0, QMAX).astype(np.int8)


def postprocess_output(acc2, q):
    """Bias add for the last layer. Output stays int32 (the logits)."""
    return (acc2 + q["b2q"]).astype(np.int32)


def int8_forward(Xq, q):
    """Fast NumPy INT8 reference forward pass. Returns (int32 logits, int8 hidden)."""
    hq = postprocess_hidden(int_matmul(Xq, q["W1q"]), q)
    return postprocess_output(int_matmul(hq, q["W2q"]), q), hq


def save_npz(q, Xq_test, y_test, X_test, path=NPZ_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, Xq_test=Xq_test, y_test=y_test, X_test=X_test, **q)


def load_npz(path=NPZ_PATH):
    """Load the saved INT8 model. Returns (q dict, Xq_test, y_test)."""
    d = np.load(path)
    q = {k: d[k] for k in ["W1q", "W2q", "b1q", "b2q"]}
    for k in ["s_in", "s_h", "s_w1", "s_w2", "m1", "out_scale"]:
        q[k] = float(d[k])
    return q, d["Xq_test"], d["y_test"]


def build_and_save():
    """Train, quantize, evaluate, and save. Returns a dict of key numbers."""
    X_train, X_test, y_train, y_test = model_mod.load_data()
    p = model_mod.train_float(X_train, y_train)
    q = quantize_model(p, X_train)
    Xq_test = quantize_input(X_test, q["s_in"])
    float_acc = model_mod.accuracy(model_mod.float_forward(X_test, p)[0], y_test)
    int8_acc = model_mod.accuracy(int8_forward(Xq_test, q)[0], y_test)
    save_npz(q, Xq_test, y_test, X_test)
    return {"float_acc": float_acc, "int8_acc": int8_acc, "q": q}


if __name__ == "__main__":
    r = build_and_save()
    print(f"float test accuracy: {r['float_acc']:.4f}")
    print(f"INT8  test accuracy: {r['int8_acc']:.4f}")
    print(f"drop: {100 * (r['float_acc'] - r['int8_acc']):.2f} percentage points")
    q = r["q"]
    print(f"scales: s_in={q['s_in']:.5f} s_w1={q['s_w1']:.5f} s_h={q['s_h']:.5f} s_w2={q['s_w2']:.5f}")
    print(f"saved {NPZ_PATH}")
