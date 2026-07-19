# rfdetr_proofs

A small Lean4/Mathlib project that formally verifies the algebraic
"primitive decomposition" tricks `../../src/ops.cpp` and `../../src/loss.cpp`
use to give ggml ops with no backward-pass case (`CLAMP`, and the elementwise
`max`/`min` used inside GIoU) a backward-capable equivalent built from ops
that do have one (`RELU`/`ADD`/`SUB`). See `RfdetrProofs/Basic.lean` for the
full set of theorems and `../../docs/decisions/0004-formal-verification.md`
for why this exists and what it covers.

## Build

```sh
lake build RfdetrProofs
```

First build fetches and caches Mathlib (`lake update`, a few minutes);
subsequent builds are incremental.
