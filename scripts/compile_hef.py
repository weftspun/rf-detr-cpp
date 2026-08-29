#!/usr/bin/env python3
"""Quantise a translated HAR and compile it to a HEF, inside the DFC image.

Normalization is compiled in, so the input layer takes uint8.
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


#: Layer types that apply a constant elementwise. A model that normalizes inside itself
#: arrives as one of these rather than as `normalization`, which is why matching only the
#: latter passed MoGe while catching VoxHammer.
CONST_ARITHMETIC = ("normalization", "ew_sub", "ew_div", "ew_mult", "ew_add", "scalar_mult")


def fed_by(L, name):
    return {k: v for k, v in L.items() if name in (v.get("input") or [])}


def pre_normalized(L, name):
    """Anything applying constants straight onto the input, whatever the translator called it."""
    return {k: v.get("type") for k, v in fed_by(L, name).items()
            if v.get("type") in CONST_ARITHMETIC}


def input_layer(L, assume_unnormalized=False):
    names = [k for k, v in L.items() if v.get("type") == "input_layer"]
    if len(names) != 1:
        sys.exit("FAIL  expected exactly one input layer, found %s" % names)
    found = pre_normalized(L, names[0])
    if found and not assume_unnormalized:
        sys.exit("FAIL  %s already applies constants to the input (%s); folding normalization "
                 "would apply it twice. Pass --assume-unnormalized to override."
                 % (names[0], ", ".join("%s=%s" % kv for kv in sorted(found.items()))))
    return names[0]


def self_test():
    """A guard that has never refused a normalized graph has not shown it can refuse one."""
    base = {"in1": {"type": "input_layer", "input": []},
            "conv1": {"type": "conv", "input": ["in1"]}}
    if input_layer(dict(base)) != "in1":
        sys.exit("FAIL  a clean graph was refused")

    planted = 0
    for kind in CONST_ARITHMETIC:
        L = dict(base, norm1={"type": kind, "input": ["in1"]})
        if pre_normalized(L, "in1") != {"norm1": kind}:
            sys.exit("FAIL  a %s on the input was not seen" % kind)
        planted += 1

    # The case that motivated this: normalization inside the module, arriving as Sub then Div.
    L = dict(base, sub1={"type": "ew_sub", "input": ["in1"]},
             div1={"type": "ew_div", "input": ["sub1"]})
    if not pre_normalized(L, "in1"):
        sys.exit("FAIL  an in-module Sub/Div on the input was not seen")

    deep = dict(base, norm9={"type": "ew_sub", "input": ["conv1"]})
    if pre_normalized(deep, "in1"):
        sys.exit("FAIL  arithmetic downstream of a conv is not pre-normalization")

    print("self-test: %d constant-applying layer types refused on the input, "
          "in-module Sub/Div refused, downstream arithmetic allowed" % planted)
    return 0


def normalization_script(L, name):
    import re
    used = [int(m.group(1)) for k in L for m in [re.search("normalization([0-9]+)", k)] if m]
    m = ", ".join("%.3f" % (v * 255) for v in MEAN)
    s = ", ".join("%.3f" % (v * 255) for v in STD)
    return "input_normalization%d = normalization([%s], [%s], %s)\n" % (
        max(used or [0]) + 1, m, s, name)


def calibration_script(batch):
    """Batch 8 exceeded a 30 GiB container; batch 1 is the same frames, lower peak."""
    return "model_optimization_config(calibration, batch_size=%d)\n" % batch


def precision_script(mode):
    # 16 bits is affordable at 25 M parameters and leaves little damage to repair.
    return "quantization_param({*}, precision_mode=%s)\n" % mode if mode else ""


def finetune_script(level, batch, epochs):
    """Forced: under 1024 frames the flow drops to 1, skipping fine-tuning."""
    out = ""
    if level:
        out += "model_optimization_flavor(optimization_level=%d)\n" % level
    args = ["finetune", "policy=enabled"]
    if batch:
        args.append("batch_size=%d" % batch)
    if epochs:
        args.append("epochs=%d" % epochs)
    if batch or epochs:
        out += "post_quantization_optimization(%s)\n" % ", ".join(args)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("har", nargs="?")
    ap.add_argument("--calib")
    ap.add_argument("--out")
    ap.add_argument("--arch", default="hailo10h")
    ap.add_argument("--calib-batch", type=int, default=1)
    ap.add_argument("--opt-level", type=int, default=0)
    ap.add_argument("--finetune-batch", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--precision", default="")
    ap.add_argument("--assume-unnormalized", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not (a.har and a.calib and a.out):
        sys.exit("FAIL  har, --calib and --out are required unless --self-test")

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
    script = (normalization_script(L, input_layer(L, a.assume_unnormalized))
              + calibration_script(a.calib_batch)
              + precision_script(a.precision)
              + finetune_script(a.opt_level, a.finetune_batch, a.epochs))
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
