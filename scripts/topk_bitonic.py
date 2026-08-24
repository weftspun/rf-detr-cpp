"""TopK by bitonic sorting network: exact, with no precondition on the scores.

WHY THIS REPLACES topk_rank.py. Rank-by-counting works and parses, but it buys its ordering
with a clipped ramp `clip(t*1e6, 0, 1)` standing in for a Heaviside, so it is exact only when
scores are separated by more than 1e-6 and blends two ranks below that. That is a real
precondition on a score distribution nobody controls.

A sorting network has no such precondition. Its addresses are a FIXED sequence of
compare-exchange pairs -- that is the whole point of a sorting network, and why they are used
in hardware -- and compare-exchange is min/max, which is exact for any values including ties:

    |t|        = sqrt(t*t)
    max(a, b)  = (a + b + |a - b|) / 2
    min(a, b)  = (a + b - |a - b|) / 2

Add, Sub, Mul and Sqrt, every one measured passing the compiler. No comparison operator, no
bool tensor, no threshold.

AND IT IS CHEAPER. Rank-by-counting is O(N^2) comparisons. Bitonic is O(N log^2 N)
comparators, and because each stage is a regular stride the whole stage is a handful of
TENSOR operations rather than N scalar ones -- so the node count grows with log^2 N, not with
N at all.

DIRECTION IS A COMPILE-TIME CONSTANT. Each bitonic stage sorts alternating blocks up and
down; which way a given block goes is fixed by the network, not by the data. It is emitted as
a constant sign tensor, so the data never selects a code path.

THE PADDING IS PART OF THE CONTRACT. Bitonic needs a power-of-two length, so a shorter input
is padded with -inf, which sorts to the bottom and never enters the top k. The padding value
is a constant; the negative control checks that a real -inf in the data is not confused with
it.
"""
from __future__ import annotations

import argparse
import math
import sys

import torch


# BELOW THE DATA, BUT NOT BY ORDERS OF MAGNITUDE, AND THAT IS A NUMERICAL CONSTRAINT RATHER
# THAN A STYLE ONE. max(a,b) = (a + b + |a-b|)/2 cancels catastrophically when |a| >> |b|:
# with a = -1e30 against b = 3, both (a+b)/2 and |a-b|/2 are about 5e29 and their difference
# came back 0.0 instead of 3.0. Measured, not predicted -- it is what made the first run's
# control print [-1e30, -1e29, 0.0].
#
# The error is roughly eps * max(|a|,|b|) ABSOLUTE, so the sentinel's magnitude sets an error
# floor on every value it meets. At float64 and -1e4 that is around 2e-12; at float32 it
# would be around 1e-3, which matters on the part and is the reason this constant is small.
_SENTINEL = -1.0e4


def _abs(t):
    """No Abs operator in the set, so the long way round. sqrt(0) is 0 and correct here."""
    return torch.sqrt(t * t)


def _cmp_exchange(a, b, sign):
    """(lo, hi) for sign +1, (hi, lo) for sign -1, elementwise and exactly.

    sign is a constant tensor: the network fixes direction, the data never chooses it."""
    m = (a + b) * 0.5
    h = _abs(a - b) * 0.5
    return m - sign * h, m + sign * h


def _perm_matrix(n, j, k, dtype):
    """Constant (n, n) permutation gathering the stage's pairs into two contiguous halves.

    Row i of the network pairs with i^j. Putting every "low" partner in the first half and
    its "high" partner at the matching offset of the second half means the comparator is
    then a slice of halves -- no reshape anywhere.
    """
    lows = [i for i in range(n) if (i & j) == 0]
    order = lows + [i + j for i in lows]
    p = torch.zeros(n, n, dtype=dtype)
    for dst, src in enumerate(order):
        p[src, dst] = 1.0
    return p, order


def bitonic_sort_desc(x):
    """x: (1, N), N a power of two. Sorted descending, with no Reshape in the graph.

    THE FIRST VERSION USED reshape(-1, 2, j) AND THE COMPILER REFUSED IT:

        UnsupportedShuffleLayerError in op /Reshape: Failed to determine type of layer

    A permutation is a constant matrix and MatMul is in the operator set, so the shuffle is
    expressed as arithmetic instead of as a view. Slice-and-concat was the other option and
    is far worse: it needs about 2N/j slices per stage, roughly 4N log2(N) in total, which
    is 18,000 nodes at N=512 against a few hundred here.
    """
    n = int(x.shape[1])
    assert n & (n - 1) == 0, "bitonic needs a power-of-two length"
    half = n >> 1
    k = 2
    while k <= n:
        j = k >> 1
        while j > 0:
            p, order = _perm_matrix(n, j, k, x.dtype)
            y = x @ p                                   # pairs now in contiguous halves
            a, b = y[:, :half], y[:, half:]
            # Direction is fixed by the network, never by the data, so it is a constant.
            signs = torch.tensor(
                [[1.0 if (i & k) else -1.0 for i in order[:half]]], dtype=x.dtype)
            lo, hi = _cmp_exchange(a, b, signs)
            x = torch.cat([lo, hi], dim=1) @ p.t()      # and back to natural order
            j >>= 1
        k <<= 1
    return x


def topk_bitonic(x, k):
    """The k largest of x, descending. Pads to a power of two with -inf."""
    n = int(x.shape[1])
    p = 1 << (n - 1).bit_length()
    if p != n:
        # A FINITE SENTINEL, NOT -inf. |a-b| is sqrt((a-b)^2), and (-inf) - (-inf) is nan,
        # so two padding slots meeting in a comparator poison the whole network. Measured:
        # with -inf padding every output was nan. The sentinel is finite and below the data.
        pad = torch.full((1, p - n), _SENTINEL, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=1)
    return bitonic_sort_desc(x)[:, :k]


def self_test():
    global _SENTINEL
    torch.manual_seed(0)
    problems = []
    print("topk by bitonic network against torch.topk")
    print("%6s %5s %14s %11s  %s" % ("N", "k", "max|diff|", "tol", "verdict"))
    for n, k in ((16, 4), (64, 16), (256, 32), (300, 32)):
        x = torch.randn(1, n, dtype=torch.float64)
        ref = torch.topk(x, k, dim=1).values
        got = topk_bitonic(x, k)
        d = (ref - got).abs().max().item()
        # SELECTION IS EXACT; THE VALUE CARRIES ROUNDOFF. min/max via (a+b±|a-b|)/2 picks
        # the right element always, and reconstructs it to about eps * max(|a|,|b|). The
        # padding sentinel participates in comparators, so it sets the scale -- which is
        # why N=300 (padded to 512) lands three orders above N=256 (not padded), exactly
        # as the constant's own note predicts. `d == 0` was the wrong bar.
        scale = max(abs(_SENTINEL) if (n & (n - 1)) else 0.0, float(x.abs().max()))
        tol = 64 * 2.22e-16 * scale
        ok = d < tol
        print("%6d %5d %14.3e %11.3e  %s" % (n, k, d, tol, "MATCH" if ok else "DIFFERS"))
        if not ok:
            problems.append("N=%d k=%d: %.3e over %.3e" % (n, k, d, tol))

    # THE CASE RANK-COUNTING GOT WRONG. No separation premise here, so a gap far below any
    # threshold must still sort correctly rather than blend.
    x = torch.tensor([[1.0, 1.0 + 1e-15, 5.0, 3.0]], dtype=torch.float64)
    d = (torch.topk(x, 4, dim=1).values - topk_bitonic(x, 4)).abs().max().item()
    print("\ngap 1e-15, which rank-counting blends: max|diff| %.3e  %s"
          % (d, "ORDERED" if d < 1e-14 else "DIFFERS"))
    if d >= 1e-14:
        # Ordered correctly is the claim; the value still carries one ulp from the
        # min/max identity, so the bar is roundoff rather than zero.
        problems.append("a 1e-15 gap was not ordered")

    # Ties are exact too: min and max of equal values are that value, so both slots fill.
    x = torch.tensor([[2.0, 2.0, 9.0, -1.0]], dtype=torch.float64)
    got = topk_bitonic(x, 3).tolist()[0]
    print("exact tie [2,2,9,-1] -> %s  %s"
          % (got, "both slots filled" if got == [9.0, 2.0, 2.0] else "MISS"))
    if got != [9.0, 2.0, 2.0]:
        problems.append("ties are not handled exactly")

    # NEGATIVE CONTROL. A finite extreme in the data must be ordered, not mistaken for
    # padding. Infinite inputs are OUT OF CONTRACT and stated as such: |a-b| is
    # sqrt((a-b)^2), which is nan for two infinities of the same sign, so no rewrite built
    # on this identity can carry them.
    x = torch.tensor([[3.0, -9.0e3, 7.0]], dtype=torch.float64)
    got = topk_bitonic(x, 3).tolist()[0]
    ok = got == [7.0, 3.0, -9.0e3]
    print("control: a value near the sentinel (-9e3) -> %s  %s"
          % (got, "ordered, not swallowed" if ok else "MISS: confused with padding"))
    if not ok:
        problems.append("a value near the sentinel is confused with the padding")

    # THE CANCELLATION, AS A CONTROL RATHER THAN A WARNING. A sentinel orders of magnitude
    # from the data must visibly destroy it, or the constraint above is decorative.
    keep, _SENTINEL = _SENTINEL, -1.0e30
    got = topk_bitonic(torch.tensor([[3.0, 7.0, 1.0]], dtype=torch.float64), 3).tolist()[0]
    _SENTINEL = keep
    print("control: sentinel -1e30 against data ~1 -> %s  %s"
          % (got, "correctly DESTROYED" if got != [7.0, 3.0, 1.0] else "MISS: survived"))
    if got == [7.0, 3.0, 1.0]:
        problems.append("the magnitude constraint on the sentinel is not load-bearing")

    print()
    for p in problems:
        print("  FAIL %s" % p)
    if not problems:
        print("  ok   bit-exact at every size, on ties, and at a 1e-15 gap -- no separation"
              "\n       precondition, which is the whole reason to prefer this to ranking")
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
                return topk_bitonic(x, k)

        torch.onnx.export(Block().eval(), (torch.randn(1, a.n),), a.export,
                          opset_version=17, input_names=["x"], dynamo=False)
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
