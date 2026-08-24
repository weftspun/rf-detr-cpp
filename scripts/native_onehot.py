"""The compiler has a one-hot layer. Every arithmetic reconstruction of one was wasted work.

WHAT THE GUIDE SAYS, AND IT WAS NEVER READ. Table 3 of the Hailo Dataflow Compiler 5.3.0 user
guide, "Supported ONNX Layers", carries a row:

    torch.nn.functional.one_hot    OneHot    Convolution with delta activation, axis=-1 only

So `OneHot` is a supported operator, realised on the accelerator as a convolution whose
activation is a delta -- parsed, per the same guide, from `val * sign(abs(x))`. It is exact by
construction in hardware.

WHAT THIS REPLACES, STATED PLAINLY BECAUSE IT IS THE WHOLE POINT. The tent kernel
`relu(1 - sqrt(t*t))` is exactly a one-hot at integer positions, which is true and was proved,
and it survives quantisation badly because `idx - i` spans the slot range while the kernel
reads only |t| < 1. Measured against a float reference with updates spanning 1..52, worst
error over the input-regime catalogue:

    int8,  tent                                   28.77
    int8,  tent normalised                        19.53
    int8,  tent normalised, kernel^8               0.26
    int16, tent normalised, kernel^8               2.2e-06
    native OneHot                                  exact, no quantisation of an index at all

Four fixes were tried before reading the table: clipping the argument (exactly identity, and
it changed nothing, because quantisation happens where `idx - i` is CREATED); sharpening to
`tent^n`; factoring the one-hot positionally into two digits (defeated by `floor` being
discontinuous -- an index 1e-12 below a block boundary lands in the wrong block, and
quantisation makes that routine rather than rare); and a 108-configuration brute force over
width, symmetry, power and radix. The brute force did find something real -- sharpening works,
but only with normalisation and only where the intermediates have headroom -- and all of it is
moot against a hardware primitive.

THE INDEX IS NOT QUANTISED HERE, WHICH IS THE ACTUAL FIX. `OneHot` consumes an INTEGER index.
The whole failure mode was an integer forced through a quantised floating-point datapath so
that a smooth kernel could pick it out again. Handing the integer to a layer that takes
integers removes the problem rather than shrinking it.

WHAT IS STILL OPEN, AND IT IS NOT SMALL. `OneHot` needs an index to consume, and in the
exported keypoint graph that index comes from `TopK`, which appears ZERO times in the 10,470
lines of the guide and is genuinely unsupported. So the elimination rewrite in
`topk_elimination.py` is still needed to PRODUCE the index; what this file removes is the
arithmetic reconstruction of the one-hot that CONSUMES it.

`axis=-1 only` is a real restriction and is asserted below rather than assumed.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F


def scatter_native(base, idx, updates, n_out):
    """base (N, n_out), idx (N, K) integer-valued, updates (N, K) -> (N, n_out).

    The one-hot is built on the LAST axis, which is the only axis the compiler supports, and
    the reduction is over K. Both `Mul` and `ReduceSum` are in Table 3."""
    oh = F.one_hot(idx.long(), n_out).to(updates.dtype)      # (N, K, n_out), one-hot on -1
    assert oh.shape[-1] == n_out, "one-hot must land on the last axis"
    return base + (updates.unsqueeze(-1) * oh).sum(dim=-2)


def gather_native(data, idx, axis_len):
    """data (A, C), idx (B, C) integer-valued -> out (B, C) = data[idx[b, c], c]."""
    oh = F.one_hot(idx.long(), axis_len).to(data.dtype)      # (B, C, A), one-hot on -1
    assert oh.shape[-1] == axis_len, "one-hot must land on the last axis"
    return (oh * data.t().unsqueeze(0)).sum(dim=-1)


def self_test():
    bad = 0
    g = torch.Generator().manual_seed(0)

    print("scatter: native OneHot vs torch scatter, exact or it is not a replacement")
    N, K, n_out = 3, 52, 104
    idx = torch.randint(0, n_out, (N, K), generator=g)
    upd = torch.randn(N, K, generator=g, dtype=torch.float64)
    base = torch.zeros(N, n_out, dtype=torch.float64)
    ref = base.clone().scatter_add_(1, idx, upd)
    got = scatter_native(base, idx.double(), upd, n_out)
    d = float((got - ref).abs().max())
    print("  %-46s max |diff| %.3g" % ("random indices with collisions", d))
    bad += d != 0.0

    print("\ngather: native OneHot vs torch.gather")
    A, C, B = 104, 7, 5
    data = torch.randn(A, C, generator=g, dtype=torch.float64)
    gi = torch.randint(0, A, (B, C), generator=g)
    d = float((gather_native(data, gi.double(), A) - torch.gather(
        data.unsqueeze(0).expand(B, A, C), 1, gi.unsqueeze(1)).squeeze(1)).abs().max())
    print("  %-46s max |diff| %.3g" % ("random indices", d))
    bad += d != 0.0

    print("\nnegative control: a FRACTIONAL index must not silently interpolate.")
    print("  the tent kernel interpolates by design; OneHot truncates via .long(), so a")
    print("  fractional index is a DIFFERENT answer and the caller must not pass one.")
    fi = torch.tensor([[1.5]], dtype=torch.float64)
    dat = torch.tensor([[0.], [10.], [20.]], dtype=torch.float64)
    v = float(gather_native(dat, fi, 3)[0, 0])
    print("  index 1.5 into [0,10,20] -> %.2f  (truncated to slot 1, NOT interpolated to 15)"
          % v)
    bad += v != 10.0

    print("\n%d check(s) wrong" % bad)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--export", metavar="PATH")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.export:
        class M(torch.nn.Module):
            def forward(self, base, idx, updates):
                return scatter_native(base, idx, updates, 104)
        torch.onnx.export(
            M().eval(),
            (torch.zeros(1, 104), torch.zeros(1, 52), torch.randn(1, 52)),
            a.export, opset_version=17, dynamo=False,
            input_names=["base", "idx", "updates"], output_names=["out"])
        import onnx
        m = onnx.load(a.export)
        ops = sorted({n.op_type for n in m.graph.node})
        print("exported %s  %d nodes" % (a.export, len(m.graph.node)))
        print("op types: %s" % ", ".join(ops))
        print("OneHot present: %s" % ("YES" if "OneHot" in ops else "NO -- it was decomposed"))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
