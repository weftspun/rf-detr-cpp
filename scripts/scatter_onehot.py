"""ScatterND with static addresses, using the tent kernel already proved for GridSample.

THE PROBLEM, MEASURED. The exported keypoint model carries 85 ScatterND nodes and all 85
take a DATA-dependent index, so they do not fold. They are the largest single group of
operators the Hailo Dataflow Compiler refuses.

THE REWRITE IS THE SAME KERNEL AS THE DEFORMABLE ONE. A scatter into a known-size output is
a sum of one-hot columns, and at integer positions the bilinear tent IS the one-hot:

    tent(idx - i) = 1 when i = idx, and 0 for every other integer i

which is `tent_of_mem_unit` at d = 0 together with `tent_int_eq_zero`, both already proved in
`weftspun/lean-deform-exact`. So

    out[i] = sum_j updates[j] * tent(idx[j] - i)

The index has moved out of the ADDRESS and into a WEIGHT, exactly as it did for GridSample,
and `i` now runs over a fixed range known at compile time.

WHAT IT COSTS. One multiply-accumulate per (output slot x update) instead of one indexed
write. For the keypoint head the output is small -- num_keypoint_classes * num_kp slots --
which is why this is worth doing here and would not be worth doing against a feature map.

THE PRICE OF EXACTNESS IS AN INTEGER PRECONDITION. tent gives a true one-hot only when the
index is an integer. A fractional index would silently interpolate between two slots rather
than failing, so the negative control below plants one.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F


def tent(t):
    """Same kernel as deform_bounded.py. |t| via sqrt(t*t) because Abs is not in the
    operator set; no eps, since sqrt(0) is 0 and correct in a forward pass."""
    return F.relu(1.0 - torch.sqrt(t * t))


def scatter_onehot(base, idx, updates, n_out):
    """base: (N, n_out). idx: (N, K) integer-valued. updates: (N, K). Returns base with
    updates written at idx, computed with static addresses only."""
    # NO BROADCAST AND NO Expand. Two earlier shapes both failed at the compiler rather
    # than in the maths. Implicit rank-3 broadcasting gave
    #
    #     ValueError: operands could not be broadcast together with shapes (104,) (52,)
    #
    # and making it explicit was worse: torch lowers `.expand` to Expand plus
    # ConstantOfShape/Equal/Where scaffolding, and the parse died on `/Expand_1`.
    #
    # The output slot is a COMPILE-TIME CONSTANT, so it never needed to be a tensor
    # dimension at all. One small block per slot, each operating on (N,K) with a scalar
    # subtrahend, and a concat at the end. n_out blocks instead of one rank-3 tensor --
    # more nodes, and every one of them an operator the per-operator run measured passing.
    cols = []
    for i in range(n_out):
        w = tent(idx - float(i))                        # (N, K), scalar subtrahend
        cols.append((updates * w).sum(dim=1, keepdim=True))
    return base + torch.cat(cols, dim=1)


def reference_scatter(base, idx, updates, n_out):
    """What the exported graph does today: an indexed accumulate."""
    out = base.clone()
    for b in range(base.shape[0]):
        for k in range(idx.shape[1]):
            out[b, int(idx[b, k].item())] += updates[b, k]
    return out


def self_test():
    torch.manual_seed(0)
    problems = []
    print("scatter via one-hot tent against an indexed accumulate")
    print("%6s %5s %14s %12s  %s" % ("n_out", "K", "max|diff|", "tol", "verdict"))
    for n_out, K in ((23, 17), (104, 52), (208, 104)):
        base = torch.randn(2, n_out, dtype=torch.float64)
        # Distinct slots per row: an indexed accumulate and a one-hot sum agree on repeats
        # too, but distinct indices are the head's actual case and keep the test readable.
        idx = torch.stack([torch.randperm(n_out)[:K] for _ in range(2)]).to(torch.float64)
        upd = torch.randn(2, K, dtype=torch.float64)
        ref = reference_scatter(base, idx, upd, n_out)
        got = scatter_onehot(base, idx, upd, n_out)
        d = (ref - got).abs().max().item()
        tol = 64 * 2.22e-16 * K
        ok = d < tol
        print("%6d %5d %14.3e %12.3e  %s" % (n_out, K, d, tol, "MATCH" if ok else "DIFFERS"))
        if not ok:
            problems.append("n_out=%d K=%d: %.3e over %.3e" % (n_out, K, d, tol))

    # NEGATIVE CONTROL. The one-hot property needs an INTEGER index. A fractional one
    # interpolates between neighbouring slots instead of failing, which is the quiet way
    # this rewrite would be wrong, so it must be caught rather than assumed away.
    base = torch.zeros(1, 8, dtype=torch.float64)
    idx = torch.tensor([[3.5]], dtype=torch.float64)
    upd = torch.tensor([[1.0]], dtype=torch.float64)
    got = scatter_onehot(base, idx, upd, 8)
    split = float(got[0, 3]), float(got[0, 4])
    spread = abs(split[0] - 1.0)
    print("\nnegative control: index 3.5 -> slots (3,4) = (%.2f, %.2f)  %s"
          % (split[0], split[1], "correctly SPLIT" if spread > 0.1 else
             "MISS: behaved as an integer index"))
    if spread <= 0.1:
        problems.append("the integer precondition is not load-bearing")

    print()
    for p in problems:
        print("  FAIL %s" % p)
    if not problems:
        print("  ok   exact to float64 accumulation at integer indices, and visibly wrong at"
              "\n       a fractional one -- so the precondition is real, not decorative.")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--export")
    ap.add_argument("--n-out", type=int, default=104)
    ap.add_argument("--k", type=int, default=52)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.export:
        n_out, K = a.n_out, a.k

        class Block(torch.nn.Module):
            def forward(self, base, idx, updates):
                return scatter_onehot(base, idx, updates, n_out)

        args = (torch.zeros(1, n_out), torch.arange(K, dtype=torch.float32).view(1, K),
                torch.randn(1, K))
        torch.onnx.export(Block().eval(), args, a.export, opset_version=17,
                          input_names=["base", "idx", "updates"], dynamo=False)
        import collections
        import onnx
        ops = collections.Counter(n.op_type for n in onnx.load(a.export).graph.node)
        print("exported %s: %d nodes, ops %s" % (a.export, sum(ops.values()), dict(ops)))
        bad = [o for o in ("ScatterND", "GatherElements", "GridSample") if o in ops]
        if bad:
            print("  FAIL %s survived the rewrite" % ", ".join(bad))
            return 1
        print("  ok   no ScatterND, GatherElements or GridSample in the graph")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
