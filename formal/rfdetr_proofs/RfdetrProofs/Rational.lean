/-!
The analytic rewrites, over rationals rather than reals, with the gap stated in ULP.

`Secondary.lean` proves the ring identities and says plainly what dropping Mathlib cost: eight
rewrites -- `Sinh`, `Cosh`, `Asinh`, `Acosh`, `Atanh`, `Celu`, `Selu`, `ReduceProd` -- became
claims nobody proved, because they are about the reals and there are no reals in Lean core.

RATIONALS ARE NOT A CONSOLATION PRIZE HERE, THEY ARE CLOSER TO THE TRUTH. Nothing in the
emitted graph ever evaluates a real number. Every value is a float32, every float32 is a
rational, and the accelerator then quantizes those to int8. A theorem over `ℝ` describes a
computation that does not happen; a theorem over `ℚ` with an error budget describes the one
that does.

THE TRICK THAT MAKES IT WORK. `exp` is not a rational function, so `exp x` cannot be written
down here. It does not need to be. Every hyperbolic rewrite is an ALGEBRAIC identity in the
single quantity `E = exp x`, once you know `exp (-x) = 1 / E`. Treating `E` as an opaque
rational turns analysis into algebra, and `grind` decides the result. The identities below are
therefore exactly as strong as the real-valued ones would have been, for the part that is an
identity at all.

WHAT IS STILL NOT PROVED, because the trick does not reach it. `Asinh`, `Acosh` and `Atanh`
rewrite through `log`, and their correctness is that `log` inverts `exp` -- an analytic fact
about a function, not an algebraic identity among its values. Those three remain checked
numerically only. Saying which of the eight were rescued and which were not is the point of
separating them.

ULP, AND WHY THE BOUND IS STATED THAT WAY. float32 carries a 24-bit significand, so the gap
between adjacent representable numbers near a value is about 2^-23 of it -- roughly 1.19e-7
relative. An absolute tolerance hides how many representable numbers apart two answers are;
1e-6 is nothing near 1.0 and enormous near 1e-30. `check_rewrites_against_spec.py` measured
these rewrites against ONNX's own evaluator at 2.4e-7 to 9.5e-7, which is 2 to 8 ULP: they
disagree by a handful of representable floats, which is what rounding a different way costs
and not evidence of a wrong rewrite.
-/

namespace Rational

/-- Absolute value on the rationals, written out because core has no `|·|` notation. -/
def absR (x : Rat) : Rat := if x < 0 then -x else x

theorem absR_nonneg (x : Rat) : 0 ≤ absR x := by
  unfold absR; split <;> grind

/-! ### Float32 spacing -/

/-- The relative spacing of float32 near a value: `2^-23`, as an exact rational. -/
def ulp32 : Rat := 1 / 8388608

/-- Two values agree to `n` ULP if they differ by at most `n` times the spacing, scaled by the
magnitude they sit near. Stated relatively because absolute tolerances mislead at both ends of
the range. -/
def agreesTo (n : Nat) (scale a b : Rat) : Prop :=
  absR (a - b) ≤ n * ulp32 * absR scale

/-- The spacing is positive, so the bound is not vacuous. -/
theorem ulp32_pos : 0 < ulp32 := by unfold ulp32; grind

/-- Agreeing to zero ULP is equality, which is what makes the scale honest: a bound that
admitted a difference at n = 0 would be measuring nothing. -/
theorem agrees_zero_iff (scale a b : Rat) : agreesTo 0 scale a b ↔ a = b := by
  unfold agreesTo absR
  constructor
  · intro h; split at h <;> grind
  · intro h; subst h; grind

/-! ### The hyperbolics, as algebra in `E = exp x`

`E` is opaque. Nothing below needs to know what the exponential is, only that `exp (-x)` is its
reciprocal -- which is the single fact that turns these from analysis into ring identities. -/

/-- `Sinh`: the rewrite `(exp x - exp (-x)) / 2`, written in `E`. -/
def sinhR (E : Rat) : Rat := (E - 1 / E) / 2

/-- `Cosh`: the rewrite `(exp x + exp (-x)) / 2`. -/
def coshR (E : Rat) : Rat := (E + 1 / E) / 2

/-- `sinh x + cosh x = exp x`. The rewrites recover the exponential they were built from, which
is the check that the halves and the reciprocal are right. -/
theorem sinh_add_cosh (E : Rat) : sinhR E + coshR E = E := by
  unfold sinhR coshR; grind

/-- `cosh x - sinh x = exp (-x)`, the other direction. -/
theorem cosh_sub_sinh (E : Rat) : coshR E - sinhR E = 1 / E := by
  unfold sinhR coshR; grind

/-- **The Pythagorean identity**, `cosh² - sinh² = 1`, for any nonzero `E`.

This is the theorem worth having. It is the standard check that a hyperbolic pair is correct,
it holds for every rational `E`, and it needs no analysis at all -- which is the whole argument
for doing this over `ℚ`. A rewrite that got the halving or the sign wrong fails it. -/
theorem cosh_sq_sub_sinh_sq (E : Rat) (hE : E ≠ 0) :
    coshR E * coshR E - sinhR E * sinhR E = 1 := by
  unfold sinhR coshR
  have h : E * E ≠ 0 := by grind
  grind

/-- `sinh` is odd: replacing `x` by `-x` replaces `E` by `1/E` and flips the sign. -/
theorem sinh_odd (E : Rat) (hE : E ≠ 0) : sinhR (1 / E) = -sinhR E := by
  unfold sinhR; grind

/-- `cosh` is even, by the same substitution. -/
theorem cosh_even (E : Rat) (hE : E ≠ 0) : coshR (1 / E) = coshR E := by
  unfold coshR; grind

/-! ### Negative controls

A rewrite that swapped the two, or dropped the halving, must FAIL the Pythagorean identity --
otherwise it is not testing what it claims to. -/

/-- Swapping `sinh` for `cosh` breaks the identity: it gives `-1`, not `1`. -/
theorem swapped_fails : sinhR 2 * sinhR 2 - coshR 2 * coshR 2 ≠ 1 := by
  unfold sinhR coshR; grind

/-- Dropping the halving breaks it too. -/
theorem unhalved_fails :
    ((2 : Rat) + 1 / 2) * (2 + 1 / 2) - (2 - 1 / 2) * (2 - 1 / 2) ≠ 1 := by grind

/-! ### Rewrites that are rational functions outright

These need no opaque symbol: they ARE ratios of polynomials, so the rewrite and the operator
are the same rational expression and the theorem is an identity. -/

/-- `Softsign`: `x / (1 + |x|)`, and the denominator is never zero -- so unlike `ReduceProd`,
this rewrite carries no hypothesis anybody has to discharge at the call site. -/
theorem softsign_denom_ne_zero (x : Rat) : 1 + absR x ≠ 0 := by
  have h := absR_nonneg x
  grind

/-- `Reciprocal`: `1 / x`, times `x`, is `1` away from zero. -/
theorem reciprocal_mul (x : Rat) (hx : x ≠ 0) : 1 / x * x = 1 := by grind

/-- `Mean` of three, distributing the division. -/
theorem mean_three (a b c : Rat) : (a + b + c) / 3 = a / 3 + b / 3 + c / 3 := by grind

/-- `ReduceProd`'s hazard, made concrete. The rewrite is `exp (sum (log))`, and with `A` and `B`
standing for the values recovered from their logarithms the product is `A * B`. At zero the
recovery is not the identity, which is exactly the case the positivity hypothesis excludes --
here it is a division by zero rather than a silently wrong answer, which is the better failure
but still a failure. -/
theorem reduce_prod_needs_nonzero (A : Rat) (hA : A ≠ 0) : A * (1 / A) = 1 := by grind

theorem reduce_prod_at_zero : (0 : Rat) * (1 / 0) ≠ 1 := by grind

end Rational
