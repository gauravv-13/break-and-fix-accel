"""Phase 2: targeted progressive bit search (the Bit-Flip Attack idea, Rakin et al., ICCV 2019).

This is a simulation on our own small model. Each round:
  1. Compute the gradient of the loss with respect to every INT8 weight, on a small
     "attacker batch" of images. Rounding/ReLU are handled with the usual
     straight-through trick (treat rounding as identity when taking the gradient).
  2. In each layer, take the top-k weights with the largest |gradient|.
  3. For every one of those candidates, try flipping each of its 8 bits and measure the
     real INT8 loss on the attacker batch.
  4. Keep the single flip that raises the loss the most. Repeat.
Stop when test accuracy is <= stop_acc (near 10% chance) or after max_flips flips.

The attacker batch comes from the TRAINING set (the attacker has a few sample images);
accuracy is measured on the separate test set.
"""
import numpy as np

from src import faults, quantize


def softmax_xent(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-np.mean(logp[np.arange(len(y)), y])), np.exp(logp)


def int8_loss(Xq, y, w):
    logits, _ = quantize.int8_forward(Xq, w)
    return softmax_xent(logits.astype(np.float64) * w["out_scale"], y)[0]


def weight_gradients(Xq, y, w):
    """d(loss)/d(integer weight) for W1q and W2q (straight-through estimate)."""
    x = Xq.astype(np.float64) * w["s_in"]
    W1 = w["W1q"].astype(np.float64) * w["s_w1"]
    W2 = w["W2q"].astype(np.float64) * w["s_w2"]
    z1 = x @ W1 + w["b1q"] * w["s_in"] * w["s_w1"]
    _, hq = quantize.int8_forward(Xq, w)
    h = hq.astype(np.float64) * w["s_h"]                 # the real (quantized) hidden values
    logits = h @ W2 + w["b2q"] * w["out_scale"]
    _, prob = softmax_xent(logits, y)
    d_logits = prob.copy()
    d_logits[np.arange(len(y)), y] -= 1.0
    d_logits /= len(y)
    dW2 = h.T @ d_logits
    dz1 = (d_logits @ W2.T) * (z1 > 0)
    dW1 = x.T @ dz1
    return {"W1q": dW1 * w["s_w1"], "W2q": dW2 * w["s_w2"]}


def run_bfa(w, X_attack, y_attack, X_test, y_test, top_k=10, max_flips=100, stop_acc=0.11):
    """Run the progressive bit search. Returns a dict with the flip positions and curves."""
    def test_acc(ww):
        return float(np.mean(np.argmax(quantize.int8_forward(X_test, ww)[0], axis=1) == y_test))

    positions, accs, losses = [], [test_acc(w)], [int8_loss(X_attack, y_attack, w)]
    cur = w
    while len(positions) < max_flips and accs[-1] > stop_acc:
        grads = weight_gradients(X_attack, y_attack, cur)
        best = (-np.inf, None)
        for layer, key in enumerate(faults.WEIGHT_KEYS):
            g = np.abs(grads[key]).ravel()
            for flat in np.argsort(-g, kind="stable")[:top_k]:
                k, n = np.unravel_index(flat, grads[key].shape)
                for bit in range(8):
                    pos = faults.position_of(layer, k, n, bit, cur)
                    loss = int8_loss(X_attack, y_attack, faults.flip_bits(cur, [pos]))
                    if loss > best[0]:
                        best = (loss, pos)
        cur = faults.flip_bits(cur, [best[1]])
        positions.append(int(best[1]))
        losses.append(best[0])
        accs.append(test_acc(cur))
    return {"positions": positions, "test_acc": accs, "attack_loss": losses, "final_weights": cur}
