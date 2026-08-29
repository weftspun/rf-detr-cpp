#!/usr/bin/env python3
"""Quantise a translated HAR and compile it to a HEF. Linux x86-64, inside the DFC image.

Normalization is compiled in rather than baked into the calibration set, so the input
layer takes uint8 -- `rfdetr`'s own ImageNet constants, scaled to 0-255.
"""
from __future__ import annotations

import argparse
import sys
import traceback

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def layers(runner):
    hn = runner.get_hn()
    if not isinstance(hn, dict):
        import json
        hn = json.loads(hn)
    return hn["layers"]


def input_layer(L):
    """The HN's own input layer name. Guessing it produces a parser error at optimize time."""
    names = [k for k, v in L.items() if v.get("type") == "input_layer"]
    if len(names) != 1:
        sys.exit("FAIL  expected exactly one input layer, found %s" % names)
    if any(v.get("type") == "normalization" for k, v in L.items()
           if any(names[0] in i for i in (v.get("input") or []))):
        sys.exit("FAIL  the input is already normalized; adding another would double-apply")
    return names[0]


def normalization_script(L, name):
    """A directive name the transformer's own normalization1..N layers have not taken."""
    import re
    used = [int(m.group(1)) for k in L for m in [re.search("normalization([0-9]+)", k)] if m]
    m = ", ".join("%.3f" % (v * 255) for v in MEAN)
    s = ", ".join("%.3f" % (v * 255) for v in STD)
    return "input_normalization%d = normalization([%s], [%s], %s)\n" % (
        max(used or [0]) + 1, m, s, name)


def calibration_script(batch):
    """Statistics collection at batch 8 exceeded a 30 GiB container; batch 1 is the same
    frames at a lower peak."""
    return "model_optimization_config(calibration, batch_size=%d)\n" % batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("har")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="hailo10h")
    ap.add_argument("--calib-batch", type=int, default=1)
    a = ap.parse_args()

    import numpy as np
    from hailo_sdk_client import ClientRunner

    calib = np.load(a.calib)
    print("calibration: %s %s" % (calib.shape, calib.dtype))
    if calib.dtype != np.uint8:
        sys.exit("FAIL  calibration is %s; the input layer is uint8" % calib.dtype)
    if calib.ndim != 4 or calib.shape[3] != 3:
        sys.exit("FAIL  calibration must be NHWC with 3 channels, got %s" % (calib.shape,))

    runner = ClientRunner(har=a.har, hw_arch=a.arch)
    L = layers(runner)
    script = normalization_script(L, input_layer(L)) + calibration_script(a.calib_batch)
    print("model script:\n%s" % script.rstrip())
    runner.load_model_script(script)

    print("optimize: quantising against %d frames..." % calib.shape[0])
    runner.optimize(calib)
    opt_har = a.out.rsplit(".", 1)[0] + ".optimized.har"
    runner.save_har(opt_har)
    print("  ok    %s" % opt_har)

    print("compile: building HEF...")
    hef = runner.compile()
    with open(a.out, "wb") as fh:
        fh.write(hef)

    # A compile that writes an empty or absent file has produced nothing, whatever it printed.
    import os
    size = os.path.getsize(a.out)
    if size == 0:
        sys.exit("FAIL  wrote a zero-byte HEF")
    print("  ok    %s  %.1f MB" % (a.out, size / 1e6))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)
