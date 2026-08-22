#!/usr/bin/env python3
"""Gate: does the real Dataflow Compiler take the device half?

Runs on Linux x86-64, because that is the only platform the DFC wheel is built for.
It needs no accelerator: parsing and emulation are software, so this answers the
go/no-go before any device is bought.

WHY IT IS NOT MERELY A COMPILE. `gate_onnx_device.py` decides the same question on
macOS from a hand-maintained allowlist, `DEVICE_OPS`. That allowlist is a claim about
this compiler, and only this script can check it. So the two are run as a PAIR and
their disagreement is the finding:

    macOS says PASS, DFC parses          -- the allowlist held
    macOS says PASS, DFC rejects op X    -- DEVICE_OPS is too generous; move X out
    macOS says FAIL on X, DFC takes X    -- DEVICE_OPS is too strict; move X in

A run that reports only the compiler's verdict throws that away, so this prints the
comparison and exits non-zero when the two disagree, even if the parse itself passed.

Run on the Fly machine:
    fly ssh console -a weftspun-hailo-dfc -C \
        "/opt/dfc/bin/python /work/scripts/gate_dfc_parse.py /work/models/backbone_576.onnx"
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import traceback


def onnx_ops(path):
    import onnx
    return collections.Counter(n.op_type for n in onnx.load(path).graph.node)


def parse(path, arch, name="device_half"):
    """(ok, detail). `ok` is whether the DFC accepted the graph."""
    from hailo_sdk_client import ClientRunner
    runner = ClientRunner(hw_arch=arch)
    try:
        runner.translate_onnx_model(path, name)
        har = path.rsplit(".", 1)[0] + f".{arch}.har"
        runner.save_har(har)
        return True, har
    except Exception as exc:            # noqa: BLE001 - the message is the measurement
        return False, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--arch", default="hailo10h")
    ap.add_argument("--expect", choices=("pass", "fail"), default="pass",
                    help="what gate_onnx_device.py predicted for this graph")
    ap.add_argument("--device-ops", help="JSON list from the macOS gate, to cross-check")
    a = ap.parse_args()

    ops = onnx_ops(a.onnx)
    print(f"DFC parse: {a.onnx}  arch={a.arch}")
    print(f"  graph: {sum(ops.values())} nodes, {len(ops)} distinct operators")

    ok, detail = parse(a.onnx, a.arch)
    print(f"  parse: {'OK' if ok else 'REJECTED'}")
    print(f"  {detail}" if ok else f"  {detail[:2000]}")

    problems = []
    if ok and a.expect == "fail":
        problems.append("the DFC accepted a graph the macOS gate rejected; "
                        "DEVICE_OPS is too strict and should be widened")
    if not ok and a.expect == "pass":
        problems.append("the DFC rejected a graph the macOS gate accepted; "
                        "DEVICE_OPS is too generous and should be narrowed")

    if a.device_ops:
        allow = set(json.loads(a.device_ops))
        outside = {o: c for o, c in ops.items() if o not in allow}
        if ok and outside:
            problems.append(f"the DFC accepted operators outside DEVICE_OPS: {outside}; "
                            f"add them rather than leaving the two gates disagreeing")
        print(f"  operators outside the macOS allowlist: {outside or 'none'}")

    # A parse that emits no HAR is not a pass, whatever it printed. Rule 3.
    if ok and not detail.endswith(".har"):
        problems.append("parse reported OK but wrote no HAR, so nothing was produced")

    if problems:
        print(f"\nFAIL ({len(problems)}):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nPASS: the DFC agrees with the macOS gate")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
