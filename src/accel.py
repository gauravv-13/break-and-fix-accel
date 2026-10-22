"""Phase 1, step 4: run the whole INT8 network through the systolic array model.

Interface used by later phases:
    logits, stats = run_inference(xq, weights)
      xq      : (M, 64) int8 inputs (use quantize.quantize_input on float pixels)
      weights : the quantized model dict (as returned by quantize.quantize_model or
                quantize.load_npz). Only W1q, W2q, b1q, b2q, m1 are used.
      checker : optional weight-row checker (see defense_checksum.py)
    logits : (M, 10) int32
    stats  : {"layers": [layer1_stats, layer2_stats], "total_cycles": ...,
              "check_failures": ...}

The matrix multiplies run on the array; bias add, ReLU and requantization run in a small
"post-processing unit" after the array (the same code as the NumPy reference, so any
difference would come from the array itself). Post-processing cycles are not counted.
"""
from src import quantize, systolic


def run_inference(xq, weights, checker=None):
    layers = []
    failures = 0
    h = xq
    for i, wkey in enumerate(["W1q", "W2q"]):
        check_row, latency = None, 0
        if checker is not None:
            check_row = lambda k, nt, row, i=i: checker.check_row(i, k, nt, row)
            latency = checker.check_latency
        acc, st = systolic.matmul(h, weights[wkey], check_row=check_row, check_latency=latency)
        layers.append(st)
        failures += st["check_failures"]
        h = quantize.postprocess_hidden(acc, weights) if i == 0 else quantize.postprocess_output(acc, weights)
    stats = {"layers": layers, "total_cycles": sum(s["total"] for s in layers),
             "check_failures": failures}
    return h, stats
