"""TopK by unrolled max-elimination, with a static ramp that makes ties impossible.

FROM THE MEMO, WITH ONE SUBSTITUTION. The structure is the memo's: unroll K steps of
ReduceMax, build a one-hot mask at the argmax, read the index off it with a dot product
against a constant index vector, then push that element below the floor before the next step.
About 6K nodes rather than the O(N log^2 N) a full sort costs, which is the right trade when
K << N -- and RF-DETR selects 300 of a much larger set.

THE SUBSTITUTION IS THE MASK, AND IT IS NOT A STYLE PREFERENCE. The memo builds it with
`GreaterOrEqual` and `Where`. Both are unmeasured on this compiler: the per-operator run
reported Greater and Where REFUSED, but that run was later shown to be reading its own
harness -- a one-operator graph cannot tell "refused" from "absorbed into its neighbour". So
their status is unknown rather than bad, and an unknown is not something to build on when an
alternative is already measured passing.

    mask = clip(1 + diff / eps, 0, 1)        diff = X - max(X), so diff <= 0

At the argmax diff is 0 and the mask is 1. Everywhere else diff <= -eps and the mask is 0.
Add, Div, Clip -- Clip and Add are in the operator set and ReduceMax was measured passing in
the per-operator run. No comparison operator and no bool tensor for the compiler to carry.

THE RAMP IS WHY THIS IS EXACT WHERE RANK-COUNTING WAS NOT. topk_rank.py needed the scores to
be separated by more than 1e-6 and blended two ranks below that -- a precondition on a
distribution nobody controls. The memo's ramp CREATES the separation instead of assuming it:

    X' = X + I * eps                         I = [0, 1, ... N-1], constant

Equal scores become unequal by at least eps, ordered by index, which is a deterministic
tie-break rather than an arbitrary one. eps then has to be small enough not to reorder
genuinely distinct scores and large enough to survive the arithmetic; both bounds are
measured in the self-test rather than asserted.

LARGE_VAL IS 65504 ON PURPOSE, which is the memo's note and a good one: the fp16 maximum. A
larger value would overflow to inf in a half-precision pipeline, and inf poisons every
identity here -- |a-b| is sqrt((a-b)^2), which is nan for two infinities.
"""
from __future__ import annotations

import argparse
import sys

import torch

LARGE_VAL = 65504.0     # fp16 max: pushes an extracted element below the floor, no overflow


def topk_elimination(x, k, span=1e-6):
    """x: (1, N). Returns (values, indices) of the k largest, static addresses only.

    `span` is the TOTAL width of the tie-break ramp, not the per-element step. That
    distinction is the memo's one error and it is not cosmetic: the memo asks for an eps
    "smaller than the minimum delta between valid scores", but the ramp spans N*eps, so the
    real bound is eps < min_delta / N. Measured with eps = 1e-4 fixed: exact at N=300 k=32,
    and 28 of 100 indices wrong at N=1024, because 1024 * 1e-4 = 0.1 exceeds the gaps
    between adjacent order statistics up there. Parameterising the span makes the bound
    independent of N.
    """
    n = int(x.shape[1])
    idx_vec = torch.arange(n, dtype=x.dtype, device=x.device).view(1, n)
    delta = span / n
    # SUBTRACTED, NOT ADDED. An increasing ramp added to the scores makes the HIGHER index
    # win a tie; torch.topk breaks ties toward the LOWER one. Subtracting matches it, which
    # matters because torch.topk is what this is checked against.
    xk = x - idx_vec * delta
    vals, idxs = [], []
    for _ in range(k):
        v = xk.max(dim=1, keepdim=True).values                  # ReduceMax
        diff = xk - v                                           # <= 0, zero only at argmax
        mask = torch.clamp(1.0 + diff / delta, 0.0, 1.0)        # one-hot, no comparison op
        idxs.append((mask * idx_vec).sum(dim=1, keepdim=True))  # index by dot product
        vals.append(v)
        xk = xk - mask * LARGE_VAL                              # drop it for the next step
    values = torch.cat(vals, dim=1)
    indices = torch.cat(idxs, dim=1)
    return values + indices * delta, indices


def self_test():
    torch.manual_seed(0)
    problems = []
    print("topk by max-elimination against torch.topk")
    print("%6s %5s %14s %14s  %s" % ("N", "k", "max|dval|", "idx mismatches", "verdict"))
    for n, k in ((64, 16), (300, 32), (300, 100), (1024, 100)):
        x = torch.randn(1, n, dtype=torch.float64) * 10.0
        rv, ri = torch.topk(x, k, dim=1)
        gv, gi = topk_elimination(x, k)
        dv = (rv - gv).abs().max().item()
        dmis = int((ri.to(torch.float64) - gi).abs().gt(0.5).sum())
        ok = dv < 1e-9 and dmis == 0
        print("%6d %5d %14.3e %14d  %s" % (n, k, dv, dmis, "MATCH" if ok else "DIFFERS"))
        if not ok:
            problems.append("N=%d k=%d: dval %.3e, %d index mismatches" % (n, k, dv, dmis))

    # THE CASE RANK-COUNTING GOT WRONG, and the reason the ramp exists.
    x = torch.tensor([[1.0, 1.0 + 1e-12, 5.0, 3.0]], dtype=torch.float64)
    rv = torch.topk(x, 4, dim=1).values
    gv, _ = topk_elimination(x, 4, span=1e-13)
    d = (rv - gv).abs().max().item()
    print("\ngap 1e-12 at span 1e-13 (below it): max|diff| %.3e  %s"
          % (d, "ORDERED" if d < 1e-9 else "DIFFERS"))
    if d >= 1e-9:
        problems.append("a 1e-12 gap was not ordered")

    # Exact ties: the ramp makes them index-ordered, which is deterministic. Both slots must
    # be filled with the true value -- rank-counting left one empty here.
    x = torch.tensor([[2.0, 2.0, 9.0, -1.0]], dtype=torch.float64)
    gv, gi = topk_elimination(x, 3)
    got = [round(v, 9) for v in gv.tolist()[0]]
    ok = got == [9.0, 2.0, 2.0] and gi.tolist()[0] == [2.0, 0.0, 1.0]  # lower index first
    print("exact tie [2,2,9,-1] -> values %s indices %s  %s"
          % (got, gi.tolist()[0], "index-ordered" if ok else "MISS"))
    if not ok:
        problems.append("ties are not broken by index")

    # NEGATIVE CONTROL 1. eps too large must reorder genuinely distinct scores, or the
    # bound on it is decorative rather than real.
    # The ramp is SUBTRACTED, so it can only demote: the top element never moves and
    # checking it proves nothing. The reordering shows up further down.
    x = torch.tensor([[0.7, 0.8, 0.9, 1.0]], dtype=torch.float64)   # ascending
    gv, gi = topk_elimination(x, 4, span=4.0)
    order = [int(i) for i in gi.tolist()[0]]
    print("control: span=4.0 against gaps of 0.1 -> index order %s  %s"
          % (order, "correctly REORDERED" if order != [3, 2, 1, 0] else "MISS"))
    if order == [3, 2, 1, 0]:
        problems.append("an oversized span did not reorder distinct scores")

    # NEGATIVE CONTROL 2. A span too small to survive the arithmetic must fail to break a
    # tie: the ramp underflows, two elements share the argmax, and the index comes back as
    # their SUM. Values still look right, which is exactly why this is worth a control --
    # the failure is silent in the half anybody would eyeball.
    x = torch.tensor([[5.0, 5.0, 1.0]], dtype=torch.float64)
    _, gi = topk_elimination(x, 1, span=1e-320)
    # Correct here is index 0, the lower of the tied pair. When the ramp underflows both
    # share the argmax and the dot product returns 0 + 1 = 1, so the failure is a plausible
    # index rather than a nan -- which is the point of testing it.
    print("control: span=1e-320 (underflows) -> index %.1f, correct is 0  %s"
          % (gi.item(), "correctly BREAKS" if abs(gi.item()) > 1e-9
             else "MISS: resolved the tie anyway"))
    if abs(gi.item()) <= 1e-9:
        problems.append("an underflowing span still resolved a tie")

    print()
    for p in problems:
        print("  FAIL %s" % p)
    if not problems:
        print("  ok   values and INDICES both exact, ties broken by index, and both bounds"
              "\n       on eps shown load-bearing rather than assumed")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--export")
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("-k", type=int, default=32)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.export:
        k = a.k

        class Block(torch.nn.Module):
            def forward(self, x):
                v, i = topk_elimination(x, k)
                return v, i

        torch.onnx.export(Block().eval(), (torch.randn(1, a.n),), a.export,
                          opset_version=17, input_names=["x"],
                          output_names=["values", "indices"], dynamo=False)
        import collections
        import onnx
        ops = collections.Counter(n.op_type for n in onnx.load(a.export).graph.node)
        print("exported %s: %d nodes, %d distinct ops" % (a.export, sum(ops.values()), len(ops)))
        print("  ops: %s" % dict(ops))
        if "TopK" in ops:
            print("  FAIL TopK survived")
            return 1
        print("  ok   no TopK in the graph")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
