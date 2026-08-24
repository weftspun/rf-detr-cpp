"""A test that runs on one input distribution has not measured anything yet.

WHY THIS EXISTS, AND IT IS A SPECIFIC FAILURE RATHER THAN A PRINCIPLE. The int8 behaviour of
the one-hot tent was tested with indices `arange(K) * 2` and reported CLEAN. The same test
with contiguous indices reports SMEARED, mask sum 1.252 instead of 1. Nothing about the
rewrite changed between those two runs: the spacing did. At stride 2 the neighbours land
outside the kernel's support and never get the chance to leak, so the test was measuring the
test's own input rather than the kernel.

The tell was available before the second run and was not looked for: two plausible input
distributions, one verdict each, and they disagree. That disagreement IS the finding, and a
single-regime pass conceals it by construction.

SO THE RULE IS NOT "SAMPLE BETTER". Nobody knows in advance which distribution is the
adversarial one -- if they did they would have used it. The rule is to run every regime and
FAIL WHEN THEY DISAGREE, which converts a lucky pass into a visible contradiction. A test
that passes everywhere is a measurement; a test that passes somewhere is an anecdote.

REGIMES ARE NAMED, NOT INVENTED PER TEST. A helper that lets each caller supply its own
inputs reproduces the original defect one layer up, because the caller picks the easy ones
again. The catalogue below is fixed and includes the two cases that actually differed, plus
the boundary case that no hand-written input would have found: values sitting exactly on the
edge of a kernel's support, where quantisation decides which side they fall.

Usage:  python scripts/check_input_regimes.py   (runs its own controls)
"""
from __future__ import annotations

import sys

import numpy as np


def regimes(n, span):
    """The fixed catalogue. Each returns a 1-D array of n values within `span`."""
    rng = np.random.default_rng(0)
    return {
        # The case that hid the defect: gaps wide enough that neighbours miss the kernel.
        "strided": np.arange(n, dtype=np.float64) * 2.0,
        # The case that found it, and the one the real graph produces.
        "contiguous": np.arange(n, dtype=np.float64),
        # Ties and near-ties, which break anything that assumes separation.
        "clustered": np.repeat(np.arange(n // 4 + 1, dtype=np.float64), 4)[:n],
        # No structure to accidentally align with a kernel width.
        "random": rng.uniform(0, span, n),
        # On the support boundary, where quantisation decides the side. No hand-written
        # input lands here, which is exactly why it is in the catalogue.
        "on-boundary": np.arange(n, dtype=np.float64) + 1.0 - 1e-12,
        # Wide dynamic range: per-tensor quantisation spends its levels on the extremes and
        # starves the region a narrow kernel actually reads.
        "wide-range": np.concatenate([np.arange(n - 1, dtype=np.float64), [span * 10]]),
    }


def sweep(predicate, domain, n=52, span=104.0, verbose=True):
    """Run `predicate(values) -> bool` over every regime. Disagreement is a failure.

    `domain` is the set of regime names the function CLAIMS to handle, and it is required
    rather than defaulted. Running a function outside its stated precondition and calling the
    disagreement a defect is its own error -- this gate's first control did exactly that,
    reporting `clustered` and `random` as failures of the tent when they are duplicate and
    fractional indices, which scatter_onehot documents as out of contract and whose own
    negative control requires to break.

    So an out-of-contract regime is reported and excluded from the agreement test. What it is
    NOT is silently skipped: a domain that quietly shrinks until the test passes is the
    failure this whole file is written against, one level up.

    Returns (problems, verdicts). An empty problems list means every IN-CONTRACT regime
    agreed, which is the only outcome that licenses reporting a single number.
    """
    all_names = set(regimes(n, span))
    unknown = set(domain) - all_names
    if unknown:
        raise ValueError("domain names regimes that do not exist: %s" % sorted(unknown))
    verdicts = {}
    for name, vals in regimes(n, span).items():
        if name not in domain:
            verdicts[name] = "out-of-contract"
            continue
        try:
            verdicts[name] = bool(predicate(vals))
        except Exception as exc:                                  # noqa: BLE001
            # A regime that throws is not a pass and not a skip: it is unmeasured, and
            # silence here would read exactly like agreement.
            verdicts[name] = None
            if verbose:
                print("      %-14s RAISED %s" % (name, type(exc).__name__))
    if verbose:
        for k, v in verdicts.items():
            print("      %-14s %s" % (k, {True: "pass", False: "FAIL", None: "unmeasured",
                                            "out-of-contract": "n/a (out of contract)"}[v]))

    seen = {v for v in verdicts.values() if v in (True, False)}
    problems = []
    if len(domain) < 2:
        problems.append("domain declares %d regime(s); a sweep over one regime is the thing "
                        "this gate exists to stop." % len(domain))
    if None in verdicts.values():
        problems.append("%d regime(s) unmeasured: %s"
                        % (sum(1 for v in verdicts.values() if v is None),
                           ", ".join(k for k, v in verdicts.items() if v is None)))
    if len(seen) > 1:
        dissent = [k for k, v in verdicts.items() if v is False]
        problems.append(
            "regimes disagree -- %s fail while the rest pass. A result that depends on which "
            "inputs were chosen is a property of the choice, not of the code."
            % ", ".join(dissent))
    return problems, verdicts


def self_test():
    """The controls are the real case: the tent kernel at int8, which must disagree."""
    sys.path.insert(0, "scripts")
    import torch
    from scatter_onehot import tent

    def quantise(x, bits=8):
        s = np.abs(x).max() / (2 ** (bits - 1) - 1)
        return np.round(x / s) * s

    def onehot_survives_int8(vals):
        """True when the mask is still one-hot after int8 quantisation."""
        worst = 0.0
        for k in range(0, 104, 7):
            t = quantise(vals - float(k))
            worst = max(worst, float(tent(torch.tensor(t)).sum()))
        return worst <= 1.0001

    def onehot_survives_float64(vals):
        worst = 0.0
        for k in range(0, 104, 7):
            worst = max(worst, float(tent(torch.tensor(vals - float(k))).sum()))
        return worst <= 1.0001

    bad = 0
    print("control 1: the tent at int8 -- regimes MUST disagree, or the sweep is decoration")
    INTEGER_DISTINCT = {"strided", "contiguous", "on-boundary", "wide-range"}
    p, _ = sweep(onehot_survives_int8, INTEGER_DISTINCT)
    if not p:
        print("  MISS  every regime agreed, so this would not have caught the real defect")
        bad += 1
    else:
        print("  ok    %s" % p[-1][:88])

    print("\ncontrol 2: the same kernel in float64 -- regimes must AGREE")
    p, _ = sweep(onehot_survives_float64, INTEGER_DISTINCT)
    if p:
        print("  MISS  regimes disagreed where the maths is exact: %s" % p[-1][:70])
        bad += 1
    else:
        print("  ok    all regimes agree, which is what licenses a single number")

    print("\n%d control(s) wrong" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(self_test())
