"""Boolean plumbing as arithmetic, so no bool tensor reaches the compiler.

WHAT IS LEFT AND WHY. After the four indexing rewrites, 28 nodes of the exported keypoint
model still sit outside the accepted operator set, and 25 of them are boolean: Not x7,
And x5, Greater x4, Less x3, IsInf x4, IsNaN x2. All were re-run through the corrected
per-operator harness and all came back REFUSED, so these are real rather than the artefacts
the first run produced. All were traced and every one is DATA-dependent, so no INPUT is
constant-foldable. Six fold anyway for a different reason -- they are a guard rather than a
computation, and a guard that never fires is the identity. See the IsInf note below.

THE SUBSTITUTION IS THE ONE TopKElimination ALREADY USES, measured passing:

    step(t)        = clip(t / delta, 0, 1)      a ramp, 0 below, 1 above, delta wide
    Greater(a, b)  = step(a - b)
    Less(a, b)     = step(b - a)
    Not(m)         = 1 - m
    And(m, n)      = m * n
    Or(m, n)       = m + n - m*n

Clip, Sub, Add and Mul are all in the accepted set. Nothing here produces a bool tensor, and
a mask that is a float in [0,1] composes with the one-hot machinery the other rewrites use
rather than needing a separate Where.

DELTA IS A PRECONDITION, NOT A TUNING KNOB, and it is the same one the ramp in
topk_elimination.py carries. Inside delta of the boundary the ramp returns a fraction rather
than 0 or 1, so the result is a blend. Exactness therefore holds only when the compared
quantities are separated by more than delta, which the control below demonstrates by
violating.

IsInf AND IsNaN ARE COVERED CONDITIONALLY, WHICH IS NOT THE SAME AS COVERED. They cannot be
rewritten the way the rest can: both ask a question about a value that has left the reals,
and every identity in this family is arithmetic on reals -- sqrt(t*t) is nan for a nan input,
and a ramp on inf saturates without distinguishing inf from a merely large number.

But they are not computing anything. Traced, the six nodes are one `nan_to_num` guard,

    Add -> IsNaN -> Where -> Cast -> IsInf -> Cast

and a guard that never fires is the identity. Exposing every IsNaN and IsInf result as a
graph output and running the model: 0 elements fired across 6 trials, three at unit scale
and three at 50x to push the activations. So on everything tested the whole chain folds
away -- which is better than a rewrite, since it removes nodes rather than adding them.

THE FLOOR IS SIX TRIALS AND THAT IS NOT A PROOF. A guard exists because somebody expected
the case; not observing it in six samples bounds nothing about the seventh. Two honest
positions follow, and they differ in what they promise:

  * fold the guard, and state that non-finite activations are out of contract -- the same
    limitation the deformable rewrite already carries, since |a-b| is nan for two infinities;
  * or replace it with a clamp to a finite range, which handles +/-inf exactly and leaves nan
    out of contract, so it is strictly safer than folding and strictly weaker than the guard.

Neither is "IsInf is rewritten". Recorded this way because the difference decides what
happens when an activation does go non-finite on the part, where nobody is watching.
"""
from __future__ import annotations

import argparse
import sys

import torch


def step(t, delta):
    """0 below the boundary, 1 above, a ramp of width delta across it."""
    return torch.clamp(t / delta, 0.0, 1.0)


def gt(a, b, delta):
    return step(a - b, delta)


def lt(a, b, delta):
    return step(b - a, delta)


def not_(m):
    """m is a mask in [0,1]; 1 - m is its complement and stays in [0,1]."""
    return 1.0 - m


def and_(m, n):
    return m * n


def or_(m, n):
    """m + n - m*n, not clip(m+n): the latter is wrong for fractional masks, returning 1
    where both inputs are 0.5 and the correct answer is 0.75."""
    return m + n - m * n


def self_test():
    torch.manual_seed(0)
    problems = []
    delta = 1e-4
    print("boolean plumbing as arithmetic, against torch's own operators")
    print("%-10s %14s  %s" % ("op", "mismatches", "verdict"))

    a = torch.randn(4096, dtype=torch.float64)
    b = torch.randn(4096, dtype=torch.float64)
    for name, got, ref in (
        ("Greater", gt(a, b, delta), (a > b).double()),
        ("Less", lt(a, b, delta), (a < b).double()),
        ("Not", not_(gt(a, b, delta)), (~(a > b)).double()),
        ("And", and_(gt(a, b, delta), lt(a, torch.zeros_like(a), delta)),
         ((a > b) & (a < 0)).double()),
        ("Or", or_(gt(a, b, delta), lt(a, torch.zeros_like(a), delta)),
         ((a > b) | (a < 0)).double()),
    ):
        mism = int((got - ref).abs().gt(1e-9).sum())
        ok = mism == 0
        print("%-10s %14d  %s" % (name, mism, "EXACT" if ok else "DIFFERS"))
        if not ok:
            problems.append("%s: %d of %d mismatched" % (name, mism, a.numel()))

    # NEGATIVE CONTROL 1. Within delta of the boundary the ramp blends, so the separation
    # premise must be load-bearing rather than decorative.
    a2 = torch.tensor([1.0], dtype=torch.float64)
    b2 = torch.tensor([1.0 - delta / 2], dtype=torch.float64)
    v = float(gt(a2, b2, delta))
    print("\ncontrol: a-b = delta/2 -> %.2f  %s"
          % (v, "correctly BLENDS" if 0.01 < v < 0.99 else "MISS: returned a hard 0 or 1"))
    if not (0.01 < v < 0.99):
        problems.append("the separation premise is not load-bearing")

    # NEGATIVE CONTROL 2. clip(m+n) is the obvious wrong Or, and it must be visibly wrong on
    # fractional masks or the choice of m+n-mn is unjustified.
    m = torch.tensor([0.5], dtype=torch.float64)
    naive = float(torch.clamp(m + m, 0.0, 1.0))
    good = float(or_(m, m))
    print("control: Or(0.5,0.5) -> clip %.2f vs m+n-mn %.2f  %s"
          % (naive, good, "correctly DIFFER" if abs(naive - good) > 0.1 else "MISS"))
    if abs(naive - good) <= 0.1:
        problems.append("the choice of Or is not justified by the control")

    print()
    for p in problems:
        print("  FAIL %s" % p)
    if not problems:
        print("  ok   exact outside delta, blending inside it, and no bool tensor anywhere."
              "\n       IsInf and IsNaN are covered CONDITIONALLY and not by this file: they"
              "\n       are a nan_to_num guard that folds when it never fires, measured over"
              "\n       six trials, which is evidence and not a proof. See the docstring.")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--export")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.export:
        class Block(torch.nn.Module):
            def forward(self, x, y):
                d = 1e-4
                m = gt(x, y, d)
                n = lt(x, torch.zeros_like(x), d)
                return and_(m, not_(n)), or_(m, n)

        torch.onnx.export(Block().eval(), (torch.randn(1, 256), torch.randn(1, 256)),
                          a.export, opset_version=17, input_names=["x", "y"], dynamo=False)
        import collections
        import onnx
        ops = collections.Counter(n.op_type for n in onnx.load(a.export).graph.node)
        print("exported %s: %d nodes, ops %s" % (a.export, sum(ops.values()), dict(ops)))
        bad = [o for o in ("Greater", "Less", "Not", "And", "Or", "Where") if o in ops]
        if bad:
            print("  FAIL %s survived the rewrite" % ", ".join(bad))
            return 1
        print("  ok   no boolean operator in the graph")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
