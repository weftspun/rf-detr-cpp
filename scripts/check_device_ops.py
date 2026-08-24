#!/usr/bin/env python3
"""Gate: is every `DEVICE_OPS` entry backed by anything?

`gate_onnx_device.py` carries a 40-operator allowlist and promises how entries got there:
"Every entry below was observed in a device-half export that the numeric check passed; nothing
is here on the strength of a datasheet." Nothing checked that. This does.

TWO KINDS OF EVIDENCE, and an entry can have either, both, or neither.

    observed     the operator appears in an ONNX export on this desk
    documented   the vendor names it, per `hailo_ops.usda`

Neither cell is a defect by itself. Observed-and-undocumented is usually shape plumbing that a
fixed-resolution graph folds away before hardware -- real support with an invisible
precondition. `Where` is in the allowlist from a real export, and a STANDALONE `Where` is
rejected because there is nothing to fold it into. `Cast` is worse: accepted, then not
performed, returning an f32 -> s32 -> f32 round trip untruncated and wrong by 0.875.

The cell that contradicts the promise is neither-observed-nor-documented.

NOT PROOF THOSE ARE WRONG, and the floor matters: exports at resolutions other than 576 were
not retained, so "not observed here" is weaker than "never observed". What this establishes is
that the promise is unverifiable for them -- a reason to record the export each entry came
from, not to delete the entries.

Usage:
    python scripts/check_device_ops.py [<dir-with-exports>]
    python scripts/check_device_ops.py --self-test
"""
from __future__ import annotations

import collections
import glob
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent


def vendor_ops():
    """Operator names the vendor documents, read from the usda layer.

    The union of lists and sequences is COMPUTED, never stored -- a third copy is a third
    thing that can disagree with the other two.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(HERE / "hailo_ops.usda"))
    if stage is None:
        raise RuntimeError("hailo_ops.usda did not open")

    def ops_at(path):
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError("hailo_ops.usda has no prim at %s" % path)
        return set(prim.GetAttribute("ops").Get() or [])

    named = ops_at("/HailoOps/Layers") | ops_at("/HailoOps/Activations")

    patterns = {}
    for child in stage.GetPrimAtPath("/HailoOps/Patterns").GetChildren():
        patterns[child.GetName()] = list(child.GetAttribute("sequence").Get() or [])

    in_patterns = {op for seq in patterns.values() for op in seq}
    return named | in_patterns, patterns


def device_ops():
    """Read the allowlist out of the gate rather than keeping a second copy."""
    src = (HERE / "gate_onnx_device.py").read_text(encoding="utf-8")

    def grab(name):
        m = re.search(name + r"\s*=\s*\{(.*?)\n\}", src, re.S)
        return set(re.findall(r'"([A-Za-z][A-Za-z0-9_]*)"', m.group(1))) if m else set()

    return grab("DEVICE_OPS"), grab("KNOWN_BLOCKERS")


def observed(paths):
    import onnx

    seen, per_file = set(), {}
    for p in paths:
        ops = {n.op_type for n in onnx.load(p).graph.node}
        per_file[os.path.basename(p)] = ops
        seen |= ops
    return seen, per_file


def onnx_surface(opset=17):
    """Every `ai.onnx` operator existing at `opset`, read from the onnx package.

    NOT a copy of the spec. `get_all_schemas()` is the wrong call and was the first thing
    tried: it returns only each operator's LATEST schema, so anything revised after `opset`
    -- Cast at 19, Conv at 22, Reshape at 19 -- is dropped despite existing at 17. It
    reported 89 where there are 178. The history has to be walked.
    """
    import onnx.defs

    by_name = collections.defaultdict(list)
    for sch in onnx.defs.get_all_schemas_with_history():
        if sch.domain in ("", "ai.onnx"):
            by_name[sch.name].append(sch)

    return {
        n
        for n, ss in by_name.items()
        if any(s.since_version <= opset and not s.deprecated for s in ss)
    }


def classify(allow, seen, documented):
    out = collections.defaultdict(list)
    for op in sorted(allow):
        key = (
            "observed" if op in seen else "unobserved",
            "documented" if op in documented else "undocumented",
        )
        out[key].append(op)
    return out


def main(argv):
    if "--self-test" in argv:
        return self_test()

    where = argv[1] if len(argv) > 1 else "."
    paths = sorted(glob.glob(os.path.join(where, "*.onnx")) + glob.glob("build/onehot/*.onnx"))
    if not paths:
        # An unmet precondition is a FAIL. With no exports the observed axis is empty and
        # every entry looks unjustified, which reads like a finding and is not one.
        print("FAIL: no .onnx files in %r -- nothing to check the allowlist against" % where)
        return 1

    allow, blockers = device_ops()
    documented, patterns = vendor_ops()
    seen, per_file = observed(paths)

    for name, ops in sorted(per_file.items()):
        print("  %-32s %3d distinct" % (name, len(ops)))
    print("  %-32s %3d union" % ("", len(seen)))
    print()

    surface = onnx_surface(17)
    print("ai.onnx at opset 17: %d operators" % len(surface))
    for label, group in [("vendor documents", documented), ("DEVICE_OPS allows", allow)]:
        share = 100.0 * len(group & surface) / len(surface)
        print("  %-20s %3d  (%2.0f%%)" % (label, len(group & surface), share))
    stray = (allow | documented) - surface
    print("  not ai.onnx operators at all: %s" % (" ".join(sorted(stray)) or "none"))
    print()

    cells = classify(allow, seen, documented)
    for key, label in [
        (("observed", "documented"), "observed AND documented"),
        (("observed", "undocumented"), "observed, documented nowhere -- passes by folding"),
        (("unobserved", "documented"), "not observed here, but documented"),
        (("unobserved", "undocumented"), "NEITHER observed NOR documented"),
    ]:
        ops = cells.get(key, [])
        print("%-52s %2d" % (label, len(ops)))
        for i in range(0, len(ops), 8):
            print("     " + " ".join(ops[i : i + 8]))
    print()

    missing = sorted(seen - allow - blockers)
    if missing:
        print("FAIL: in an export and NOT in DEVICE_OPS: %s" % " ".join(missing))
        return 1

    without = cells.get(("unobserved", "undocumented"), [])
    if without:
        print("FAIL: %d entries have no evidence of either kind: %s"
              % (len(without), " ".join(without)))
        print("      the docstring promises none is there on a datasheet's say-so.")
        return 1

    print("ok   every DEVICE_OPS entry is observed here or documented")
    return 0


def self_test():
    """Controls. Each entry must land in the cell it belongs to, or this is decoration."""
    from pxr import Usd

    cases = [
        ("observed and documented", {"Add"}, {"Add"}, {"Add"}, ("observed", "documented")),
        ("observed, documented nowhere", {"Cast"}, {"Cast"}, set(), ("observed", "undocumented")),
        ("documented, not seen here", {"Gemm"}, set(), {"Gemm"}, ("unobserved", "documented")),
        ("neither", {"Range"}, set(), set(), ("unobserved", "undocumented")),
    ]
    print("controls:")
    bad = 0
    for label, allow, seen, doc, want in cases:
        got = classify(allow, seen, doc)
        ok = list(got.keys()) == [want]
        print("  %s %-34s -> %s" % ("ok  " if ok else "BAD ", label, list(got.keys())))
        bad += 0 if ok else 1

    documented, patterns = vendor_ops()
    stage = Usd.Stage.Open(str(HERE / "hailo_ops.usda"))
    layers = set(stage.GetPrimAtPath("/HailoOps/Layers").GetAttribute("ops").Get())
    acts = set(stage.GetPrimAtPath("/HailoOps/Activations").GetAttribute("ops").Get())

    checks = [
        ("the usda layer opens and composes", stage is not None),
        ("Erf is in neither list alone", "Erf" not in layers and "Erf" not in acts),
        ("Erf is reachable through the gelu sequence", "Erf" in patterns["Gelu"]),
        ("Erf therefore counts as documented", "Erf" in documented),
        ("Cast is documented nowhere", "Cast" not in documented),
        ("Where is documented nowhere", "Where" not in documented),
        ("Sqrt is documented, as an activation", "Sqrt" in acts),
        ("Sqrt is NOT a layer -- reading one list alone would miss it", "Sqrt" not in layers),
    ]
    for label, cond in checks:
        print("  %s %s" % ("ok  " if cond else "BAD ", label))
        bad += 0 if cond else 1

    print("\n%d control(s) wrong" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
