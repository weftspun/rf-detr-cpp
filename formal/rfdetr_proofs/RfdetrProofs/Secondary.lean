/-!
The secondary-path rewrites in `scripts/hailo_ops.usda`, proved.

The accelerator toolchain implements 58 of the 178 ONNX operators available at opset 17. The
other 120 either rewrite into operators it does implement -- the `Secondary` scope -- or carry a
reason there is none, in `Refused`. Each rewrite is an algebraic claim, and this is where the
ones expressible over a ring are settled rather than asserted.

NO MATHLIB, AND THAT IS A TRADE RATHER THAN A TIDY-UP. `grind` in Lean core decides polynomial
identities over `Int`, which covers every rewrite whose content is ring algebra: selection,
boolean, the reductions that are sums, the masks. Importing Mathlib to reach `ring` would cost
4400 modules and buy nothing for those.

WHAT IT COSTS, stated plainly because it is a real loss. `Sinh`, `Cosh`, `Asinh`, `Acosh`,
`Atanh`, `Celu`, `Selu` and `ReduceProd` are claims about the REALS -- exponentials,
logarithms, and a positivity hypothesis. There are no reals in Lean core, so those rewrites are
not proved here at all. They are instead:

  - checked numerically against the executable specification, by
    `scripts/check_rewrites_against_spec.py`, which runs ONNX's own reference evaluator and
    compares; and
  - searched for counterexamples over `Int` by `Witness.lean` where an integer analogue exists.

That is weaker than a proof and is not pretending otherwise. A numeric check over 64 sampled
points cannot rule out a disagreement at the 65th, which is exactly what a theorem would.
`ReduceProd` is the one that most deserves a proof it no longer has: its rewrite is
`exp (sum (log))`, valid only for strictly positive inputs, and a zero makes it silently wrong
rather than an error.

The trigonometric rewrites are absent for a different reason and would be absent with Mathlib
too: they are minimax polynomials, not identities, so a theorem claiming equality would be
false. They are marked APPROXIMATE in the usda.

CONDITIONS ARE `Int`-VALUED, matching what an ONNX comparison produces: 0 or 1, never a `Prop`.
-/

namespace Secondary

/-! ### Selection

`Where` matters most: the target rejects a standalone `Where` at parse, and `Sub`, `Mul` and
`Add` are all implemented. -/

/-- `Where` with the condition true. -/
theorem where_true (a b : Int) : b + (a - b) * 1 = a := by grind

/-- `Where` with the condition false. -/
theorem where_false (a b : Int) : b + (a - b) * 0 = b := by grind

/-- The ring identity underneath `Where`, true for EVERY condition rather than only `{0,1}`.
The `{0,1}` restriction constrains what the result MEANS, not whether the algebra holds. -/
theorem where_ring (a b c : Int) : b + (a - b) * c = b * (1 - c) + a * c := by grind

/-- Selection, as a case split on a condition that is genuinely 0 or 1. -/
theorem where_eq (a b c : Int) (hc : c = 0 ∨ c = 1) :
    b + (a - b) * c = if c = 1 then a else b := by
  rcases hc with h | h <;> subst h <;> grind

/-! ### Boolean algebra as arithmetic on `{0, 1}`

`boolean_arithmetic.py` is the working of these. Each is stated twice on purpose: once as an
unconditional ring identity, and once as a claim about truth values that needs its hypotheses.
Separating them says exactly where the hypotheses are load-bearing. -/

/-- `Or` by de Morgan. Unconditional. -/
theorem or_ring (a b : Int) : a + b - a * b = 1 - (1 - a) * (1 - b) := by grind

/-- `Xor` differs from `Or` by the factor of two, and only where both operands hold. -/
theorem xor_ring (a b : Int) : a + b - 2 * (a * b) = (a + b - a * b) - a * b := by grind

/-- `Not` is an involution. -/
theorem not_involutive (a : Int) : 1 - (1 - a) = a := by grind

/-- `And` on `{0,1}`. -/
theorem and_eq (a b : Bool) :
    (if a then 1 else 0) * (if b then 1 else 0) = (if a && b then (1 : Int) else 0) := by
  cases a <;> cases b <;> simp

/-- `Or` on `{0,1}`. The `- a*b` is what stops two trues giving 2. -/
theorem or_eq (a b : Bool) :
    (if a then 1 else 0) + (if b then 1 else 0)
      - (if a then 1 else 0) * (if b then 1 else 0) = (if a || b then (1 : Int) else 0) := by
  cases a <;> cases b <;> simp

/-- `Xor` on `{0,1}`. -/
theorem xor_eq (a b : Bool) :
    (if a then 1 else 0) + (if b then 1 else 0)
      - 2 * ((if a then 1 else 0) * (if b then 1 else 0))
      = (if xor a b then (1 : Int) else 0) := by
  cases a <;> cases b <;> simp

/-- `Or` and `Xor` are not interchangeable, witnessed where they diverge. `plausible` finds
this same input by search in `Witness.lean`; here it is settled. -/
theorem or_ne_xor_at_one : (1 : Int) + 1 - 1 * 1 ≠ 1 + 1 - 2 * (1 * 1) := by decide

/-! ### Comparison -/

/-- `GreaterOrEqual` and `LessOrEqual` both rest on negating a truth value. -/
theorem negate_truth (c : Bool) :
    1 - (if c then (1 : Int) else 0) = (if c then 0 else 1) := by
  cases c <;> simp

/-- `Sign`: `(x > 0) - (x < 0)`.

This is why the rewrite is not `x / |x|`: the quotient is undefined at zero, and this gives 0
there, which is what the specification requires. -/
theorem sign_eq (x : Int) :
    (if 0 < x then (1 : Int) else 0) - (if x < 0 then 1 else 0) =
      if 0 < x then 1 else if x < 0 then -1 else 0 := by
  rcases Int.lt_trichotomy x 0 with h | h | h
  · simp [h, Int.not_lt.mpr (Int.le_of_lt h)]
  · simp [h]
  · simp [h, Int.not_lt.mpr (Int.le_of_lt h)]

/-! ### Reductions and masks -/

/-- `Sum` associates, so a left fold is the sum. -/
theorem sum_assoc (a b c : Int) : a + b + c = a + (b + c) := by grind

/-- `Square`: `Pow(x, 2)` as a multiplication, avoiding a general power. -/
theorem square_eq (x : Int) : x * x = x ^ 2 := by grind

/-- `Expand`: multiplying by a constant one-tensor changes nothing. -/
theorem mask_identity (x : Int) : x * 1 = x := by grind

/-- `Trilu`: multiplying by a zero mask erases. -/
theorem mask_zero (x : Int) : x * 0 = 0 := by grind

/-- `CumSum` at three elements: the prefix sums a lower-triangular matrix of ones produces. -/
theorem cumsum_three (a b c : Int) :
    (a, a + b, a + b + c) = (a, a + b, a + (b + c)) := by grind

end Secondary
