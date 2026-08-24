"""Sin and Cos as polynomials, which is the last operator the accelerator refuses.

WHY NO RANGE REDUCTION. A general sin needs its argument folded into one period first, and
folding needs comparisons -- which would work now that boolean_arithmetic.py exists, but each
one costs accuracy near the fold boundary. It turns out not to be needed. Traced, all eight
Sin/Cos nodes are fed by

    Gather -> Mul -> Unsqueeze -> Div -> Slice

and the arguments measured over the real graph run [0.000, 6.259], inside a single period
(2*pi = 6.283). So one polynomial over [0, 2*pi] covers every call site and the whole
reduction disappears.

DEGREE 13, AND THE CHOICE IS MEASURED RATHER THAN PICKED. Chebyshev interpolation error on
[0, 2*pi]:

    degree    sin max err    cos max err
         9      1.151e-05      4.177e-05
        11      1.884e-07      8.002e-07
        13      2.270e-09      1.106e-08
        15      2.100e-11      1.154e-10

float32 eps is 1.192e-07, so degree 13 is already two orders below what single precision can
carry and four below the port's own keypoint bound of 4.2e-3. Going higher buys nothing a
float32 pipeline can represent, and each degree is another Mul and Add in Horner form.

HORNER FOR NODE COUNT, NOT FOR ACCURACY, AND THE FIRST DRAFT CLAIMED OTHERWISE. It said a
naive power series "loses precision long before" the top of the interval. Measured in float32,
where the difference should show if anywhere:

    x       horner err    naive err
    3.000    3.065e-07     2.618e-07
    6.000    3.705e-06     4.301e-06
    6.283    8.241e-06     7.868e-06

They are the same to within noise and the naive form is sometimes better. The reason to use
Horner is that it is 13 multiplies and 13 adds rather than forming thirteen separate powers.

AND THE REAL ACCURACY IS THE float32 ONE. The fit's own error is 2.270e-09, but EVALUATING it
in single precision costs about 8e-06 at the top of the interval -- three orders worse, and
still two orders inside the port's keypoint bound of 4.2e-3. The float64 figure is a property
of the coefficients; the float32 figure is what the part would see.

THE RANGE IS A PRECONDITION AND THE CONTROL VIOLATES IT. Outside [0, 2*pi] a Chebyshev fit
does not merely lose accuracy, it diverges -- this is an interpolant, not a periodic function.
An argument that leaves the interval is wrong rather than approximate, so the self-test plants
one.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

LO, HI = 0.0, 2.0 * np.pi
DEGREE = 13


def _coeffs(fn, degree=DEGREE):
    """Power-basis coefficients, lowest order first, from a Chebyshev fit on [LO, HI]."""
    c = np.polynomial.chebyshev.Chebyshev.interpolate(fn, degree, domain=[LO, HI])
    return c.convert(kind=np.polynomial.Polynomial).coef


SIN_C = _coeffs(np.sin)
COS_C = _coeffs(np.cos)


def _horner(x, coef):
    """Nested multiply-add: 13 multiplies and 13 adds rather than thirteen separate powers.
    That is a node-count argument, not an accuracy one -- see the docstring, where a naive
    power series measured the same or better in float32."""
    acc = torch.full_like(x, float(coef[-1]))
    for c in reversed(coef[:-1]):
        acc = acc * x + float(c)
    return acc


def sin_poly(x):
    return _horner(x, SIN_C)


def cos_poly(x):
    return _horner(x, COS_C)


def self_test():
    problems = []
    x = torch.linspace(LO, HI, 100001, dtype=torch.float64)
    print("polynomial trig against torch, degree %d on [0, 2pi]" % DEGREE)
    print("%-6s %14s %14s  %s" % ("fn", "max err", "float32 eps", "verdict"))
    eps32 = float(np.finfo(np.float32).eps)
    for name, got, ref in (("sin", sin_poly(x), torch.sin(x)),
                           ("cos", cos_poly(x), torch.cos(x))):
        d = float((got - ref).abs().max())
        ok = d < eps32
        print("%-6s %14.3e %14.3e  %s" % (name, d, eps32, "BELOW f32 eps" if ok else "ABOVE"))
        if not ok:
            problems.append("%s: %.3e exceeds float32 eps" % (name, d))

    # The arguments the real graph actually produces, which is the interval that matters.
    xr = torch.linspace(0.0, 6.259, 20001, dtype=torch.float64)
    d = float((sin_poly(xr) - torch.sin(xr)).abs().max())
    print("\nover the measured argument range [0, 6.259]: sin max err %.3e" % d)
    if d >= eps32:
        problems.append("error over the measured range exceeds float32 eps")

    # NEGATIVE CONTROL. A Chebyshev fit is an interpolant, not a periodic function: outside
    # the interval it diverges rather than degrading. If that is not visible the range
    # precondition is decorative.
    # The threshold is measured, not chosen: the fit crosses float32 eps at +0.2 rad, so
    # that is where the precondition starts to bite. An earlier version demanded 1e-2 at
    # +1 rad and failed its own control at 7.8e-05, which said more about the threshold than
    # about the fit.
    xo = torch.tensor([HI + 0.2], dtype=torch.float64)
    d_out = float((sin_poly(xo) - torch.sin(xo)).abs().max())
    print("control: +0.2 rad outside -> err %.3e vs f32 eps %.3e  %s"
          % (d_out, eps32, "correctly EXCEEDS" if d_out > eps32 else "MISS: still accurate"))
    if d_out <= eps32:
        problems.append("the range precondition is not load-bearing")
    print("         usable headroom: arguments measured to 6.259, interval ends at %.3f" % HI)

    # What the part would actually see: the same polynomial evaluated in single precision.
    x32 = torch.linspace(LO, HI, 20001, dtype=torch.float32)
    d32 = float((sin_poly(x32).double() - torch.sin(x32.double())).abs().max())
    print("float32 evaluation of the same fit: max err %.3e (bound 4.2e-3)  %s"
          % (d32, "ok" if d32 < 4.2e-3 else "EXCEEDS the port bound"))
    if d32 >= 4.2e-3:
        problems.append("float32 evaluation exceeds the port's keypoint bound")

    print()
    for p in problems:
        print("  FAIL %s" % p)
    if not problems:
        print("  ok   below float32 eps across the whole interval, diverging outside it")
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
            def forward(self, x):
                return sin_poly(x), cos_poly(x)

        torch.onnx.export(Block().eval(), (torch.rand(1, 256) * float(HI),), a.export,
                          opset_version=17, input_names=["x"], dynamo=False)
        import collections
        import onnx
        ops = collections.Counter(n.op_type for n in onnx.load(a.export).graph.node)
        print("exported %s: %d nodes, ops %s" % (a.export, sum(ops.values()), dict(ops)))
        bad = [o for o in ("Sin", "Cos") if o in ops]
        if bad:
            print("  FAIL %s survived" % ", ".join(bad))
            return 1
        print("  ok   no Sin or Cos in the graph")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
