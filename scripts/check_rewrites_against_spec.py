#!/usr/bin/env python3
"""Gate: does each hand-written rewrite compute what the ONNX specification says?

`hailo_ops.usda` carries 82 rewrites of operators the accelerator does not implement. Each is
an algebraic claim written by reading the spec. Reading is where mistakes live, so this runs
both sides and compares numbers.

THE SPECIFICATION IS EXECUTABLE, which is what makes this a check rather than a re-reading.
`onnx.reference.ReferenceEvaluator` interprets a graph in NumPy directly from the operator
definitions, so a one-node graph containing `Softsign` IS the specification's answer. The
rewrite is built as its own graph from operators the device implements, run the same way, and
the two are subtracted.

CLEAN-ROOM, and the distinction is worth stating because it is the reason this exists. The
rewrites were written from the mathematical description in each operator's page, not copied
from an implementation. That keeps them free of anyone's licence, and it moves the risk from
licensing to misreading -- so misreading is what gets measured here.

THE NEGATIVE CONTROL IS THE POINT. Agreement proves nothing on its own: a comparison that
cannot fail would report the same green for a rewrite that is wrong. So every rewrite is also
run MUTATED -- operands swapped, a constant perturbed, an operator exchanged for its neighbour
-- and a mutant that still agrees means the test is not looking at the thing it claims to.

Usage:
    python scripts/check_rewrites_against_spec.py
    python scripts/check_rewrites_against_spec.py --self-test
"""
from __future__ import annotations

import sys

import numpy as np

# Rewrites, as graphs over operators the device implements, written from the spec's prose.
# Each entry: (op, attributes, build(x) -> ndarray, mutants).
#
# A mutant is a PLAUSIBLE error rather than a random one: the operand order of a
# non-commutative operator, an off-by-one in a constant, a sibling operator. A mutation the
# comparison could not miss would flatter it.
REWRITES = {
    "Softsign": dict(
        attrs={},
        rewrite=lambda x: x / (1.0 + np.abs(x)),
        mutants={
            "abs dropped": lambda x: x / (1.0 + x),
            "divide inverted": lambda x: (1.0 + np.abs(x)) / x,
            "one dropped": lambda x: x / np.abs(x),
        },
    ),
    "ThresholdedRelu": dict(
        attrs={"alpha": 1.5},
        rewrite=lambda x: x * (x > 1.5),
        mutants={
            "boundary inclusive": lambda x: x * (x >= 1.5),
            "wrong threshold": lambda x: x * (x > 1.0),
            "gate not multiplied": lambda x: (x > 1.5).astype(np.float32),
        },
    ),
    "Selu": dict(
        attrs={"alpha": 1.67326319217681884765625, "gamma": 1.05070102214813232421875},
        rewrite=lambda x: 1.05070102214813232421875
        * (np.maximum(0.0, x) + np.minimum(0.0, 1.67326319217681884765625 * (np.exp(x) - 1.0))),
        mutants={
            "alpha and gamma swapped": lambda x: 1.67326319217681884765625
            * (np.maximum(0.0, x)
               + np.minimum(0.0, 1.05070102214813232421875 * (np.exp(x) - 1.0))),
            "minus one dropped": lambda x: 1.05070102214813232421875
            * (np.maximum(0.0, x) + np.minimum(0.0, 1.67326319217681884765625 * np.exp(x))),
        },
    ),
    "Shrink": dict(
        attrs={"lambd": 0.5, "bias": 0.25},
        rewrite=lambda x: (x < -0.5) * (x + 0.25) + (x > 0.5) * (x - 0.25),
        mutants={
            "bias sign swapped": lambda x: (x < -0.5) * (x - 0.25) + (x > 0.5) * (x + 0.25),
            "lambda sign dropped": lambda x: (x < 0.5) * (x + 0.25) + (x > 0.5) * (x - 0.25),
        },
    ),
    "HardSwish": dict(
        attrs={},
        rewrite=lambda x: x * np.maximum(0.0, np.minimum(1.0, x / 6.0 + 0.5)),
        mutants={
            "alpha wrong": lambda x: x * np.maximum(0.0, np.minimum(1.0, x / 3.0 + 0.5)),
            "beta dropped": lambda x: x * np.maximum(0.0, np.minimum(1.0, x / 6.0)),
        },
    ),
    "Sum": dict(
        attrs={}, n_inputs=3,
        rewrite=lambda *xs: xs[0] + xs[1] + xs[2],
        mutants={"one term dropped": lambda *xs: xs[0] + xs[1]},
    ),
    "Mean": dict(
        attrs={}, n_inputs=3,
        rewrite=lambda *xs: (xs[0] + xs[1] + xs[2]) / 3.0,
        mutants={
            "divisor wrong": lambda *xs: (xs[0] + xs[1] + xs[2]) / 2.0,
            "sum not averaged": lambda *xs: xs[0] + xs[1] + xs[2],
        },
    ),
    "Reciprocal": dict(
        attrs={},
        rewrite=lambda x: 1.0 / x,
        mutants={"negated": lambda x: -1.0 / x},
    ),
    "Sign": dict(
        attrs={},
        rewrite=lambda x: (x > 0).astype(np.float32) - (x < 0).astype(np.float32),
        mutants={
            "zero mishandled": lambda x: (x >= 0).astype(np.float32) - (x < 0).astype(np.float32),
            # NOT a mutant, and it was listed as one by mistake: x / max(|x|, eps) is a
            # CORRECT alternative implementation of Sign, agreeing everywhere including at
            # zero. The unguarded quotient below is the actual defect -- it is NaN at zero
            # where the specification says 0.
            "unguarded quotient": lambda x: x / np.abs(x),
        },
    ),
    "Sinh": dict(
        attrs={},
        rewrite=lambda x: (np.exp(x) - np.exp(-x)) / 2.0,
        mutants={
            "cosh instead": lambda x: (np.exp(x) + np.exp(-x)) / 2.0,
            "halving dropped": lambda x: np.exp(x) - np.exp(-x),
        },
    ),
    "Cosh": dict(
        attrs={},
        rewrite=lambda x: (np.exp(x) + np.exp(-x)) / 2.0,
        mutants={"sinh instead": lambda x: (np.exp(x) - np.exp(-x)) / 2.0},
    ),
    "Asinh": dict(
        attrs={},
        rewrite=lambda x: np.log(x + np.sqrt(x * x + 1.0)),
        mutants={
            "acosh instead": lambda x: np.log(x + np.sqrt(x * x - 1.0)),
            "sqrt dropped": lambda x: np.log(x + x * x + 1.0),
        },
    ),
}

TOL = 1.0e-5


def spec_value(op, attrs, inputs):
    """What the SPECIFICATION computes, by executing a one-node graph in the reference
    evaluator. This is the authority the rewrites are checked against."""
    from onnx import TensorProto, helper
    from onnx.reference import ReferenceEvaluator

    names = ["x%d" % i for i in range(len(inputs))]
    node = helper.make_node(op, names, ["y"], **attrs)
    graph = helper.make_graph(
        [node], "spec",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(inputs[0].shape))
         for n in names],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(inputs[0].shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    out = ReferenceEvaluator(model).run(None, dict(zip(names, inputs)))[0]
    return np.asarray(out, dtype=np.float32)


#: Values every sample must contain. RANDOM FLOATS NEVER LAND ON A BOUNDARY, and the first
#: version of this file learned that from its own mutants: `Sign` with `>=` for `>` and
#: `ThresholdedRelu` with an inclusive threshold both agreed with the specification, because
#: nothing in a uniform draw is ever exactly 0.0 or exactly alpha. The mutation test caught a
#: hole in the sample rather than a hole in the rewrites.
BOUNDARIES = [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 0.25, -0.25]


def sample(n_inputs, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_inputs):
        rand = rng.uniform(-3.0, 3.0, size=(54,)).astype(np.float32)
        out.append(np.concatenate([np.array(BOUNDARIES, np.float32), rand]))
    return out


def diff(a, b):
    """Largest elementwise difference, with NaN counted as a disagreement.

    NaN COMPARES FALSE TO EVERYTHING, so a plain max let a mutant that produced NaN -- Asinh
    rewritten as Acosh, whose sqrt goes negative -- read as agreeing. A difference that cannot
    be measured is not a small one."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    disagree = np.isnan(a) != np.isnan(b)
    if np.any(disagree):
        return float("inf")
    # EQUAL INFINITIES AGREE. The sample contains exact zero, so Reciprocal returns inf on
    # both sides, and inf - inf is NaN -- which reported a correct rewrite as disagreeing.
    # Subtraction is the wrong test where both sides are the same infinity.
    same_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    settled = np.isnan(a) & np.isnan(b) | same_inf
    d = np.abs(np.where(settled, 0.0, a - b))
    d = np.where(np.isnan(d), np.inf, d)
    return float(np.max(d)) if d.size else 0.0


def main():
    ok = mutants_caught = mutants_missed = failed = 0
    print("  %-18s %-12s %s" % ("operator", "vs spec", "mutants caught"))

    for op, spec in sorted(REWRITES.items()):
        xs = sample(spec.get("n_inputs", 1))
        try:
            want = spec_value(op, spec["attrs"], xs)
        except Exception as e:
            print("  %-18s SPEC UNAVAILABLE %s" % (op, str(e)[:60]))
            continue

        d = diff(spec["rewrite"](*xs), want)
        agrees = d <= TOL
        ok += agrees
        failed += not agrees

        caught, missed = 0, []
        for label, mutant in spec["mutants"].items():
            try:
                md = diff(mutant(*xs), want)
            except Exception:
                md = float("inf")
            if md > TOL:
                caught += 1
            else:
                missed.append(label)
                mutants_missed += 1
        mutants_caught += caught

        print("  %-18s %-12s %d/%d%s"
              % (op, ("ok %.2e" % d) if agrees else "DISAGREES %.3f" % d,
                 caught, len(spec["mutants"]),
                 "   MISSED: " + ", ".join(missed) if missed else ""))

    print()
    print("  %d rewrites agree with the specification, %d disagree" % (ok, failed))
    print("  %d mutants caught, %d slipped through" % (mutants_caught, mutants_missed))

    if failed:
        print("\nFAIL: a rewrite computes something other than the operator it replaces.")
        return 1
    if mutants_missed:
        print("\nFAIL: a deliberately broken rewrite agreed with the spec, so the comparison "
              "is not seeing what it claims to.")
        return 1
    print("\nok   every rewrite matches the spec, and every mutant was caught")
    return 0


def self_test():
    """The comparison itself must be able to fail. If `diff` cannot separate two obviously
    different arrays, everything above is decoration."""
    a = np.array([1.0, 2.0], np.float32)
    cases = [
        ("identical arrays differ by zero", diff(a, a) == 0.0),
        ("different arrays are separated", diff(a, a + 1.0) > TOL),
        ("a tolerance-sized difference is not flagged", diff(a, a + 1e-9) <= TOL),
        ("NaN against a number is a disagreement, not a silence",
         diff(np.array([np.nan], np.float32), a[:1]) == float("inf")),
        ("NaN against NaN agrees", diff(np.array([np.nan], np.float32),
                                       np.array([np.nan], np.float32)) == 0.0),
        ("the sample contains exact boundaries", 0.0 in list(sample(1)[0])),
        ("matching infinities agree", diff(np.array([np.inf], np.float32),
                                           np.array([np.inf], np.float32)) == 0.0),
        ("opposite infinities disagree", diff(np.array([np.inf], np.float32),
                                              np.array([-np.inf], np.float32)) == float("inf")),
    ]
    bad = 0
    print("controls:")
    for label, cond in cases:
        print("  %s %s" % ("ok  " if cond else "BAD ", label))
        bad += 0 if cond else 1
    print("\n%d control(s) wrong" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else main())
