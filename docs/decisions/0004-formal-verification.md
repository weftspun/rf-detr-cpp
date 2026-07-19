# 4. Formal verification (Lean4/Mathlib) of the backward-pass primitive tricks

* Status: accepted
* Date: 2026-07-19

## Context and Problem Statement

Phase 2 (`docs/decisions/0003-training.md`) had to work around several ggml
ops missing a backward-pass case (`GGML_OP_NORM`, `GGML_UNARY_OP_SIGMOID`,
`GGML_OP_CONCAT`, `GGML_OP_CLAMP`) by rebuilding them from primitives that DO
have backward support (`RELU`/`ADD`/`SUB`/`SET`/`EXP`/`LOG`). Each rebuild is
an algebraic identity claim -- e.g. `clamp_diff`'s
`lo + relu(x-lo) - relu(x-hi) = clamp(x, lo, hi)` -- verified so far only by
numeric diff-testing against upstream reference values on a handful of
sampled inputs (`tests/test_loss.cpp`, `tests/test_norm_backward.cpp`).
Numeric testing on samples can't rule out an identity that's subtly wrong on
some input region the samples don't cover -- exactly the failure mode that
would be silent (a real number, just occasionally the wrong one) rather than
a crash. This project's `docs/decisions/0001-open-work.md` already has two
open items (the SegXLarge divergence, the mask-head residual mechanism) that
are unexplained numeric drift with no confirmed root cause -- worth ruling
out "one of the algebraic tricks is subtly wrong for some input" as a
contributing hypothesis wherever it's cheap to do so.

How can these identities get a correctness guarantee stronger than "matched
on the samples I tried"?

## Decision

Add a small Lean4/Mathlib subproject, `formal/rfdetr_proofs/`, that formally
proves (for ALL real inputs, not sampled ones) the identities this port
actually ships:

- `elementwise_max_eq`/`elementwise_min_eq`: `a + relu(b-a) = max a b` and
  `a - relu(a-b) = min a b` (`src/loss.cpp:167-172`).
- `clamp_diff_eq`: `lo + relu(x-lo) - relu(x-hi) = clamp x lo hi` for
  `lo ≤ hi` (`src/ops.cpp:160-163`) -- this is the exact primitive keeping
  `sigmoid`'s output away from 0/1 before `log`, and `bbox_reparam_decode_diff`'s
  `delta_wh` away from overflowing `exp`, both directly load-bearing for the
  numerical-blowup fix in this port's training demo (`0003-training.md`).
- `iou_bounds`: `0 ≤ IoU ≤ 1`, derived (not assumed) from the exact
  `elementwise_max`/`min`+`relu` intersection-area construction
  `detection_loss` uses (`src/loss.cpp:256-265`), given only that both input
  boxes are valid (`x1 ≤ x2`, `y1 ≤ y2`).
- `giou_bounds_full`: **`-1 ≤ GIoU ≤ 1`**, wired directly to the same eight
  box coordinates and the same GIoU formula `detection_loss` computes
  (`src/loss.cpp:273`) -- the property that keeps the loss's aggregate
  magnitude bounded per matched box. This is the mathematical fact underlying
  why a per-element clamp doesn't automatically bound a sum over many
  elements, but this ONE term (GIoU) is bounded regardless of scale.

One hypothesis is taken as a cited external fact rather than re-derived:
`union ≤ enclose` (the smallest axis-aligned box enclosing two boxes has area
at least their set union's, since both boxes are literal subsets of the
enclosing box). This is 2D measure-theoretic (`A, B ⊆ C ⟹ vol(A∪B) ≤ vol(C)`)
and isn't algebraically reducible to the 1D interval facts the rest of the
proof uses; formalizing it in Mathlib would need `MeasureTheory.volume` over
`ℝ × ℝ` and product-measure lemmas, out of scope for what this exercise is
for. It's cited to Theorem 1 of the original GIoU paper (Rezatofighi et al.,
CVPR 2019), which proves the same fact via the same set-containment
argument -- not a novel or unverified claim, just not re-derived here.

All twelve theorems in `RfdetrProofs/Basic.lean` compile with **zero
`sorry`s and zero `axiom`s** (grep-verified) -- everything except the one
cited hypothesis above is proven from Mathlib's real-number order-lattice
lemmas (`max_eq_left`, `min_def`, `mul_le_mul`, etc.) plus `linarith`.

## What this did and didn't find

No bug was found in the shipped C++ -- every identity checked matches
exactly what `src/ops.cpp`/`src/loss.cpp` compute. This is a genuine (if
unsurprising) result: it rules out "one of these specific algebraic
rewrites is subtly wrong" as an explanation for the still-open SegXLarge
divergence and mask-head-residual items, the same way last session's
blanket-`ggml_set_output` experiment ruled out graph-allocator buffer reuse
for SegXLarge. Neither finding pinpoints the actual root cause of either
open item, but each closes off one branch of the search space with
certainty stronger than a numeric spot-check could provide.

## Why not extend this further right now

Formalizing the `union ≤ enclose` measure-theoretic fact, or verifying the
Hungarian-matching algorithm's optimality (`src/loss.cpp`'s
`kuhn_munkres`), would be the natural next steps if this direction is worth
investing in further, but both are substantially larger undertakings (real
2D measure theory; a nontrivial graph/combinatorial-optimization proof) than
the "speedrun" pace the rest of this session's loop has been operating at.
Documented here as a natural follow-up, not attempted this pass.

## Consequences

- `formal/rfdetr_proofs/` is a self-contained Lean4/Mathlib project
  (`lake build RfdetrProofs`); its `.lake/` build cache (~7GB after fetching
  Mathlib) is gitignored, only the proof source + lockfiles are committed.
- This is documentation/assurance tooling, not part of the C++ build --
  `CMakeLists.txt` is untouched, and nothing here affects `rfdetr`'s runtime
  behavior.
- Confirms (doesn't newly discover) that `clamp_diff`/`elementwise_max`/
  `elementwise_min`/GIoU-boundedness are correct as shipped; no code changes
  resulted from this exercise.
