"""TopK by counting ranks, so the last indexing blocker leaves the host too.

WHY BOTHER, GIVEN TopK IS A NATURAL CUT POINT. It is, and Hailo's own DETR is cut exactly
there. But it is also the LAST refused operator once GridSample, ScatterND and GatherElements
have compiler-accepted rewrites, and a cut is not free: it costs a USB 3.1 round trip per
frame, plus the host half of the pipeline existing at all. Converting it means the graph does
not leave the part.

SELECTION IS DATA-ORDERED, WHICH IS NOT THE SAME AS DATA-ADDRESSED. Sorting needs to know
which element is larger; it does not need to compute an address from the data. Rank by
counting does the first without the second:

    rank[i] = sum_j step(x[j] - x[i])        how many elements beat element i
    sel[m,i] = tent(rank[i] - m)             the one-hot picking rank m
    out[m]  = sum_i x[i] * sel[m,i]

Every index here is a compile-time constant. `tent` is the same kernel the other three
rewrites use, and at integer ranks it is exactly a one-hot -- proved in
`weftspun/lean-deform-exact`.

THE STEP IS WHERE THE EXACTNESS GOES, AND IT IS STATED RATHER THAN HIDDEN. A true Heaviside
is not available: `Greater` returns a bool the compiler would then have to carry, and
`t/sqrt(t*t)` is 0/0 at a tie. So step is a clipped ramp, `clip(t * BIG, 0, 1)`, which is
exact when the scores are separated by more than 1/BIG and WRONG below that -- it starts
blending two ranks. That is a real precondition on the score distribution, so the negative
control below plants a near-tie and requires it to break.

Ties are also genuinely ambiguous: two equal scores give equal ranks, both one-hots land on
the same slot, and one output rank goes empty. TopK on ties is arbitrary anyway, but "the
compiler picks" and "this returns zero" are different arbitrary, so the control covers it.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

BIG = 1.0e6


def tent(t):
    return F.relu(1.0 - torch.sqrt(t * t))


def step(t):
    """1 when t is comfortably positive, 0 when comfortably negative, a ramp between.
    Clip and Mul only -- no bool tensor for the compiler to carry."""
    return torch.clamp(t * BIG, 0.0, 1.0)


def topk_rank(x, k):
    """x: (1, N). Returns the k largest values in descending order, static addresses only."""
    n = x.shape[1]
    ranks = []
    for i in range(n):
        xi = x[:, i:i + 1]                              # static slice, compile-time index
        ranks.append(step(x - xi).sum(dim=1, keepdim=True))
    rank = torch.cat(ranks, dim=1)                      # (1, N), 0 = largest
    cols = []
    for m in range(k):
        sel = tent(rank - float(m))                     # (1, N) one-hot at rank m
        cols.append((x * sel).sum(dim=1, keepdim=True))
    return torch.cat(cols, dim=1)


def self_test():
    torch.manual_seed(0)
    problems = []
    print("topk by rank-counting against torch.topk")
    print("%5s %4s %14s %12s  %s" % ("N", "k", "max|diff|", "tol", "verdict"))
    for n, k in ((32, 8), (64, 16), (128, 32)):
        # Well-separated scores: the premise the clipped step rests on.
        x = (torch.randperm(n).to(torch.float64) + 1.0).view(1, n)
        ref = torch.topk(x, k, dim=1).values
        got = topk_rank(x, k)
        d = (ref - got).abs().max().item()
        tol = 64 * 2.22e-16 * n
        ok = d < tol
        print("%5d %4d %14.3e %12.3e  %s" % (n, k, d, tol, "MATCH" if ok else "DIFFERS"))
        if not ok:
            problems.append("N=%d k=%d: %.3e over %.3e" % (n, k, d, tol))

    # NEGATIVE CONTROL 1. Scores closer than 1/BIG blend two ranks rather than ordering
    # them, so the separation premise must be load-bearing.
    x = torch.tensor([[1.0, 1.0 + 1e-9, 5.0, 3.0]], dtype=torch.float64)
    ref = torch.topk(x, 4, dim=1).values
    d = (ref - topk_rank(x, 4)).abs().max().item()
    print("\ncontrol 1: gap 1e-9 (below 1/BIG = %.0e) -> max|diff| %.3e  %s"
          % (1.0 / BIG, d, "correctly DIFFERS" if d > 1e-3 else "MISS: agreed anyway"))
    if d <= 1e-3:
        problems.append("the separation premise is not load-bearing")

    # NEGATIVE CONTROL 2. Exact ties collapse two one-hots onto one slot and leave a rank
    # empty. TopK on ties is arbitrary, but this is a DIFFERENT arbitrary and says so.
    x = torch.tensor([[2.0, 2.0, 9.0]], dtype=torch.float64)
    got = topk_rank(x, 3)
    print("control 2: exact tie [2,2,9] -> %s  %s"
          % (got.tolist()[0], "correctly leaves a slot empty" if 0.0 in got.tolist()[0]
             else "MISS: resolved the tie somehow"))
    if 0.0 not in got.tolist()[0]:
        problems.append("the tie behaviour is not what the docstring says")

    print()
    for p in problems:
        print("  FAIL %s" % p)
    if not problems:
        print("  ok   exact on separated scores; blends below 1/BIG and empties a slot on a"
              "\n       tie, both of which are preconditions rather than surprises")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--export")
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("-k", type=int, default=16)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.export:
        k = a.k

        class Block(torch.nn.Module):
            def forward(self, x):
                return topk_rank(x, k)

        torch.onnx.export(Block().eval(), (torch.randn(1, a.n),), a.export,
                          opset_version=17, input_names=["x"], dynamo=False)
        import collections
        import onnx
        ops = collections.Counter(n.op_type for n in onnx.load(a.export).graph.node)
        print("exported %s: %d nodes, %d distinct ops" % (a.export, sum(ops.values()), len(ops)))
        print("  ops: %s" % dict(ops))
        if "TopK" in ops:
            print("  FAIL TopK survived the rewrite")
            return 1
        print("  ok   no TopK in the graph")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
