#!/usr/bin/env python3
"""Quantisation cost: one graph, two precisions, same held-out frames."""
from __future__ import annotations

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("har")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--skip", type=int, default=1024)
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--arch", default="hailo10h")
    a = ap.parse_args()

    import numpy as np
    from hailo_sdk_client import ClientRunner, InferenceContext

    frames = np.load(a.calib, mmap_mode="r")
    if frames.shape[0] < a.skip + a.n:
        sys.exit("FAIL  need %d frames, have %d" % (a.skip + a.n, frames.shape[0]))
    x = np.ascontiguousarray(frames[a.skip:a.skip + a.n])
    print("frames: %s, skipping the first %d the quantiser saw" % (x.shape, a.skip))

    runner = ClientRunner(har=a.har, hw_arch=a.arch)
    with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
        native = runner.infer(ctx, x)
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        quant = runner.infer(ctx, x)

    native = native if isinstance(native, list) else [native]
    quant = quant if isinstance(quant, list) else [quant]
    worst = 0.0
    for i, (nt, qt) in enumerate(zip(native, quant)):
        nt, qt = np.asarray(nt, np.float64), np.asarray(qt, np.float64)
        d = np.abs(nt - qt)
        scale = max(float(np.abs(nt).max()), 1e-12)
        print("  output %d %-18s max|diff| %.6e  mean %.6e  rel %.4f%%"
              % (i, str(nt.shape), d.max(), d.mean(), 100 * d.max() / scale))
        worst = max(worst, d.max() / scale)

    # A control: the same tensor against itself must differ by nothing.
    same = float(np.abs(np.asarray(native[0], np.float64)
                        - np.asarray(native[0], np.float64)).max())
    print("  control  native vs itself max|diff| %.3e (must be 0)" % same)
    if same != 0.0:
        sys.exit("FAIL  the comparison cannot tell identical tensors apart")

    print("\nworst relative error across outputs: %.4f%%" % (100 * worst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
