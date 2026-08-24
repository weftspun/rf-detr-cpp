"""Deformable sampling with static addresses and data-dependent weights.

THE PROBLEM. Deformable attention samples a feature map at reference + learned offset, and
torch lowers that to `GridSample`, which the Hailo Dataflow Compiler refuses. Measured on the
exported keypoint model, all 8 GridSample nodes take a DATA-dependent grid, so the constant
fold that removed the `Tile` blocker does not apply here.

THE REWRITE. Bilinear interpolation is a separable tent kernel, and the tent is zero outside
one pixel. So sampling at y+dy is a weighted sum over INTEGER shifts k of the feature map,

    out[y] = sum_k  V[y+k] * relu(1 - |dy - k|)

and when |dy| <= r only k in [-r-1, r+1] can carry weight. Every V[y+k] is a STATIC shift --
a pad and a slice, fixed at compile time -- and everything data-dependent has moved into the
scalar weight. No address depends on the image.

WHY THAT MATTERS HERE. The addresses were the objection, not the arithmetic. This form uses
Pad, Slice, Sub, Mul, Sqrt, Relu and Add, every one of which the per-operator run measured
the compiler accepting.

|dy - k| IS WRITTEN AS sqrt((dy-k)^2). `Abs` is not in the operator set and `Max` is not
either, so the absolute value is taken the long way round. eps keeps the gradient finite at
zero; it also biases the weight by about eps/2 at the exact knot, which is why the tolerance
below is stated rather than assumed.

THE COST IS THE POINT, NOT THE POSSIBILITY. (2r+3)^2 shifted copies per sampling location
against one gather. r is the clamp radius and the trade is measured in `self_test`, not
argued for.

Usage:  python scripts/deform_bounded.py --self-test
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F


def shift2d(padded, ky, kx, h, w, P):
    """padded is v zero-padded by P on every side, so this returns v[y+ky, x+kx] with zeros
    outside. One Pad for the whole block and a constant Slice per shift -- which is the
    claim: every address is fixed at compile time."""
    return padded[:, :, P + ky:P + ky + h, P + kx:P + kx + w]


def bounded_sample(v, dy, dx, r):
    """Bilinear sample of v at (y+dy, x+dx) with |dy|,|dx| <= r, static addresses only.

    v  : (N, C, H, W)
    dy : (N, 1, H, W) pixel offsets, likewise dx
    """
    # eps IS THE DOMINANT ERROR TERM, NOT FLOAT ACCUMULATION, AND IT WAS MEASURED. With
    # eps=1e-12 the weight at an exact knot is relu(1 - sqrt(1e-12)) = 1 - 1e-6, a
    # part-per-million bias that showed up as a 4.6e-10 relative disagreement -- far above
    # float64 roundoff and therefore not roundoff. sqrt(0) is 0 and correct in a forward
    # pass; only its derivative is undefined, and nothing here differentiates. So eps is 0
    # and the tolerance below is set from accumulation alone.
    eps = 0.0
    n, c, h, w = v.shape
    P = r + 1
    padded = F.pad(v, (P, P, P, P))
    out = torch.zeros_like(v)
    for ky in range(-P, P + 1):
        wy = F.relu(1.0 - torch.sqrt((dy - ky) ** 2 + eps))
        if not bool((wy > 0).any()):
            continue
        for kx in range(-P, P + 1):
            wx = F.relu(1.0 - torch.sqrt((dx - kx) ** 2 + eps))
            if not bool((wx > 0).any()):
                continue
            out = out + shift2d(padded, ky, kx, h, w, P) * (wy * wx)
    return out


def reference_sample(v, dy, dx):
    """What torch does today: grid_sample at reference + offset, align_corners=False."""
    n, c, h, w = v.shape
    ys = torch.arange(h, dtype=v.dtype).view(1, 1, h, 1).expand(n, 1, h, w)
    xs = torch.arange(w, dtype=v.dtype).view(1, 1, 1, w).expand(n, 1, h, w)
    py, px = ys + dy, xs + dx
    gy = (2.0 * py + 1.0) / h - 1.0
    gx = (2.0 * px + 1.0) / w - 1.0
    grid = torch.cat([gx, gy], dim=1).permute(0, 2, 3, 1)
    return F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


def self_test():
    torch.manual_seed(0)
    problems = []
    print("bounded-offset deformable sampling against grid_sample")
    print("%4s %10s %12s %12s %10s  %s"
          % ("r", "shifts", "max|diff|", "rel", "tol", "verdict"))
    for r in (1, 2, 3):
        n, c, h, w = 1, 8, 16, 16
        v = torch.randn(n, c, h, w, dtype=torch.float64)
        # Offsets strictly inside the clamp, which is the premise the rewrite rests on.
        dy = (torch.rand(n, 1, h, w, dtype=torch.float64) * 2 - 1) * r * 0.98
        dx = (torch.rand(n, 1, h, w, dtype=torch.float64) * 2 - 1) * r * 0.98
        ref = reference_sample(v, dy, dx)
        got = bounded_sample(v, dy, dx, r)
        d = (ref - got).abs().max().item()
        rel = d / ref.abs().max().item()
        shifts = (2 * r + 3) ** 2
        # A FIXED ABSOLUTE TOLERANCE ACROSS DIFFERENT TERM COUNTS IS THE WRONG BAR. r=3
        # accumulates 81 products where r=1 accumulates 25, so the floor scales with the
        # sum. 64 * eps_f64 * shifts is generous against that and still four orders tighter
        # than the port's own keypoint bound of 4.2e-3.
        tol = 64 * 2.22e-16 * shifts
        ok = d < tol
        print("%4d %10d %12.3e %12.3e %10.3e  %s"
              % (r, shifts, d, rel, tol, "MATCH" if ok else "DIFFERS"))
        if not ok:
            problems.append("r=%d: max|diff| %.3e over tol %.3e" % (r, d, tol))

    # NEGATIVE CONTROL. A rewrite that matched whatever it was handed would prove nothing,
    # so an offset outside the clamp must break it: that is the premise being load-bearing.
    r = 1
    v = torch.randn(1, 8, 16, 16, dtype=torch.float64)
    dy = torch.full((1, 1, 16, 16), 2.5, dtype=torch.float64)   # outside |dy| <= r
    dx = torch.zeros(1, 1, 16, 16, dtype=torch.float64)
    d = (reference_sample(v, dy, dx) - bounded_sample(v, dy, dx, r)).abs().max().item()
    print("\nnegative control: offset 2.5 with r=1 -> max|diff| %.3e  %s"
          % (d, "correctly DIFFERS" if d > 1e-3 else "MISS: rewrite agreed anyway"))
    if d <= 1e-3:
        problems.append("the clamp premise is not load-bearing; the control did not fire")

    print()
    for p in problems:
        print("  FAIL %s" % p)
    if not problems:
        print("  ok   exact within 1e-9 inside the clamp, and wrong outside it, which is the"
              "\n       whole claim: the addresses are static and the premise is real.")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--export", help="write an ONNX graph of the rewrite here")
    ap.add_argument("--radius", type=int, default=2)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.export:
        r = a.radius

        class Block(torch.nn.Module):
            def forward(self, v, dy, dx):
                return bounded_sample(v, dy, dx, r)

        v = torch.randn(1, 8, 16, 16)
        dy = torch.rand(1, 1, 16, 16) * 2 - 1
        dx = torch.rand(1, 1, 16, 16) * 2 - 1
        torch.onnx.export(Block().eval(), (v, dy, dx), a.export, opset_version=17,
                          input_names=["v", "dy", "dx"], dynamo=False)
        import collections
        import onnx
        ops = collections.Counter(n.op_type for n in onnx.load(a.export).graph.node)
        print("exported %s: %d nodes, ops %s" % (a.export, sum(ops.values()), dict(ops)))
        if "GridSample" in ops:
            print("  FAIL GridSample survived the rewrite")
            return 1
        print("  ok   no GridSample in the graph")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
