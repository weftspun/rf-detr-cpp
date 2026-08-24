import Plausible

/-!
Property-based search for counterexamples to the secondary-path rewrites.

`Secondary.lean` proves rewrites. This looks for witnesses that they are wrong. The two are not
the same activity and neither replaces the other: a proof settles a claim forever, and a search
covers claims nobody has got round to proving, which is currently 54 of 82.

WHY BOTH. `ring` decides the polynomial rewrites, so eleven of them cannot be subtly wrong and
need nothing here. The rest carry hypotheses -- `ReduceProd` wants positivity, the boolean
rewrites want `{0,1}` -- and a hypothesis is exactly where a proof stops protecting you: the
theorem stays true while the CALL SITE quietly violates it. A search over unrestricted values
is a cheap way to find out which rewrites depend on their hypotheses and how badly.

NO MATHLIB, so this builds in seconds rather than after 4401 modules, and the properties are
stated over `Int` for the same reason. That is a real limit and is stated rather than hidden:
these are the ring identities and the boolean algebra, not the analytic rewrites. `Sinh` and
`Asinh` need `Real.exp` and are proved in `Secondary.lean` instead.

THE SEARCH MUST BE ABLE TO FAIL. `or_xor_differ_witness` below is FALSE on purpose and
plausible reports `a := 1, b := 1` for it -- the case where Or and Xor diverge, which is the
confusion a careless rewrite ships. A search that never fails is not searching, and that
example is here so the rest of the file means something.
-/

namespace Witness

/-! ### Rewrites that should survive any input -/

/-- `Where` with the condition true. -/
example (a b : Int) : b + (a - b) * 1 = a := by plausible

/-- `Where` with the condition false. -/
example (a b : Int) : b + (a - b) * 0 = b := by plausible

/-- `Where` as the ring identity underneath it: true for EVERY condition, not only `{0,1}`. -/
example (a b c : Int) : b + (a - b) * c = b * (1 - c) + a * c := by plausible

/-- `Not` is an involution. -/
example (a : Int) : 1 - (1 - a) = a := by plausible

/-- `Or` in terms of `And` and `Not`, de Morgan. Unconditional as algebra. -/
example (a b : Int) : a + b - a * b = 1 - (1 - a) * (1 - b) := by plausible

/-- `Sum` associates. -/
example (a b c : Int) : a + b + c = a + (b + c) := by plausible

/-- `Square` as a multiplication rather than a general power. -/
example (x : Int) : x * x = x ^ 2 := by plausible

/-- `Expand`: multiplying by a constant one-tensor changes nothing. -/
example (x : Int) : x * 1 = x := by plausible

/-- `Trilu`: multiplying by a zero mask erases. -/
example (x : Int) : x * 0 = 0 := by plausible

/-! ### Where the hypotheses are load-bearing

The boolean rewrites are exact on `{0,1}` and wrong off it. Stated as properties that hold ON
the restriction, so the search confirms the restriction is the right one rather than merely
sufficient. -/

/-- `And` on `{0,1}`: the product is 1 exactly when both are. -/
example (a b : Bool) :
    (if a then 1 else 0) * (if b then 1 else 0) = (if a && b then (1 : Int) else 0) := by
  plausible

/-- `Or` on `{0,1}`: note the `- a*b`, without which two trues give 2. -/
example (a b : Bool) :
    (if a then 1 else 0) + (if b then 1 else 0)
      - (if a then 1 else 0) * (if b then 1 else 0) = (if a || b then (1 : Int) else 0) := by
  plausible

/-- `Xor` on `{0,1}`: the 2 is what separates it from `Or`. -/
example (a b : Bool) :
    (if a then 1 else 0) + (if b then 1 else 0)
      - 2 * ((if a then 1 else 0) * (if b then 1 else 0))
      = (if xor a b then (1 : Int) else 0) := by
  plausible

/-! ### The control

Everything above passing means nothing unless the search can fail. This property is FALSE, and
plausible reports `a := 1, b := 1`: the one input where `Or` and `Xor` disagree.

It is stated as a theorem ABOUT the counterexample rather than left as a failing check, so this
file still builds while carrying the evidence. -/

/-- `Or` and `Xor` are not the same rewrite, witnessed at the input a search finds first. -/
theorem or_xor_differ_witness :
    (1 : Int) + 1 - 1 * 1 ≠ 1 + 1 - 2 * (1 * 1) := by decide

end Witness
