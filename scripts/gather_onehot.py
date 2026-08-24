"""GatherElements with static addresses, from the same tent kernel as the other two rewrites.

THE LAST OF THE INDEXING BLOCKERS BAR ONE. The exported keypoint model carries 85 ScatterND,
8 GridSample and 2 GatherElements, and all of them take a DATA-dependent index. The first two
have rewrites that the Hailo Dataflow Compiler accepts; this is the third.

    out[i, j] = data[idx[i, j], j]                      an indexed read
              = sum_k data[k, j] * tent(idx[i, j] - k)  a weighted sum over a fixed range

At integer positions the bilinear tent is a one-hot, so the two agree exactly. `k` runs over
the gathered axis, whose length is known at compile time, and the index has moved from the
address into a multiplier.

WHY IT IS THE CHEAP ONE. The gathered axis here is short, so the fixed range is short. That
is what decides whether this rewrite is worth making: cost is (axis length) multiply-
accumulates per output element, which is fine against a query dimension and would not be
against a feature map.

NO BROADCAST, AND THAT WAS LEARNED THE HARD WAY. The scatter rewrite failed twice at the
compiler before it failed at nothing: implicit rank-3 broadcasting is refused outright, and
`.expand` lowers to Expand plus ConstantOfShape/Equal/Where scaffolding which is refused too.
The gathered index is a compile-time constant, so it is emitted as one block per position
with a scalar subtrahend and a concat at the end.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F


def tent(t):
    """|t| via sqrt(t*t): Abs is not in the operator set. No eps -- sqrt(0) is 0 and correct
    in a forward pass, and an eps of 1e-12 was measured biasing the deformable rewrite by a
    part per million at the knot."""
    return F.relu(1.0 - torch.sqrt(t * t))


def gather_onehot(data, idx, axis_len):
    """data: (A, C). idx: (B, C) integer-valued, indexing the first axis of data.
    Returns out[i, j] = data[idx[i, j], j], with every address fixed at compile time."""
    out = torch.zeros(idx.shape, dtype=data.dtype, device=data.device)
    for k in range(axis_len):
        w = tent(idx - float(k))                        # (B, C), scalar subtrahend
        out = out + data[k].unsqueeze(0) * w
    return out


def reference_gather(data, idx):
    """What the exported graph does today, along axis 0."""
    return torch.gather(data, 0, idx.long())


def self_test():
    torch.manual_seed(0)
    problems = []
    print("gather via one-hot tent against torch.gather")
    print("%6s %5s %14s %12s  %s" % ("axis", "cols", "max|diff|", "tol", "verdict"))
    for A, C in ((17, 8), (104, 16), (300, 4)):
        data = torch.randn(A, C, dtype=torch.float64)
        idx = torch.randint(0, A, (6, C)).to(torch.float64)
        ref = reference_gather(data, idx)
        got = gather_onehot(data, idx, A)
        d = (ref - got).abs().max().item()
        tol = 64 * 2.22e-16 * A
        ok = d < tol
        print("%6d %5d %14.3e %12.3e  %s" % (A, C, d, tol, "MATCH" if ok else "DIFFERS"))
        if not ok:
            problems.append("A=%d C=%d: %.3e over %.3e" % (A, C, d, tol))

    # NEGATIVE CONTROL. The one-hot needs an INTEGER index; a fractional one interpolates
    # between neighbours instead of failing, which is the quiet way this would be wrong.
    data = torch.tensor([[0.0], [10.0], [20.0]], dtype=torch.float64)
    got = gather_onehot(data, torch.tensor([[1.5]], dtype=torch.float64), 3)
    print("\nnegative control: index 1.5 into [0,10,20] -> %.2f  %s"
          % (float(got[0, 0]),
             "correctly INTERPOLATED" if abs(float(got[0, 0]) - 15.0) < 1e-6
             else "MISS: behaved as an integer index"))
    if abs(float(got[0, 0]) - 15.0) >= 1e-6:
        problems.append("the integer precondition is not load-bearing")

    print()
    for p in problems:
        print("  FAIL %s" % p)
    if not problems:
        print("  ok   exact at integer indices, interpolating at a fractional one")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--export")
    ap.add_argument("--axis-len", type=int, default=104)
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.export:
        A = a.axis_len

        class Block(torch.nn.Module):
            def forward(self, data, idx):
                return gather_onehot(data, idx, A)

        args = (torch.randn(A, a.cols),
                torch.randint(0, A, (6, a.cols)).float())
        torch.onnx.export(Block().eval(), args, a.export, opset_version=17,
                          input_names=["data", "idx"], dynamo=False)
        import collections
        import onnx
        ops = collections.Counter(n.op_type for n in onnx.load(a.export).graph.node)
        print("exported %s: %d nodes, ops %s" % (a.export, sum(ops.values()), dict(ops)))
        bad = [o for o in ("GatherElements", "ScatterND", "GridSample") if o in ops]
        if bad:
            print("  FAIL %s survived the rewrite" % ", ".join(bad))
            return 1
        print("  ok   no GatherElements, ScatterND or GridSample in the graph")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
