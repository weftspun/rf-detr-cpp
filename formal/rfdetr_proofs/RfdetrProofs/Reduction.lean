/-!
Every operator reduces to ones the device can execute.

`scripts/hailo_ops.usda` records 58 operators the accelerator implements and 82 rewrites of
operators it does not. The question that matters for emission is not whether any single rewrite
is correct -- `Secondary.lean` handles that -- but whether the table BOTTOMS OUT: a rewrite
naming an operator that is itself unsupported, whose rewrite names another, is only useful if
that chain reaches primitives instead of running forever or looping.

`check_device_ops.py` computes the least fixpoint and reports 0 stuck across 82 rewrites, so
139 of the 178 operators at opset 17 reach the executable set. That is a measurement of one
table on one day. This file is the argument for WHY such a check is sufficient, which is the
part that does not need re-running.

NO MATHLIB. The claim is about a finite rewrite relation and needs none of it; importing it
would cost 4388 modules to prove something about `Nat`. `Secondary.lean` imports Mathlib
because real analysis genuinely needs it, and this file is a deliberate contrast.

REVIEWED AGAINST TorchLean (github.com/nktkt/leanx, MIT), which formalises neural networks in
Lean 4 for IBP and CROWN robustness bounds. It was not reusable here: its `ONNXOp` is a
twelve-constructor subset -- gemm, relu, sigmoid, tanh, conv, batchNorm, the pools, flatten,
dropout, add, matMul -- chosen for bound propagation over a fixed network, and its own comment
says it currently supports Gemm, ReLU, Sigmoid and Tanh. Bounding a network's output is a
different question from reducing an operator set, and borrowing its representation would have
imported a vocabulary that cannot express the 178 operators this has to cover.
-/

namespace Reduction

-- Operators are opaque. Nothing here depends on what an operator computes, only on which
-- operators its rewrite mentions, so a concrete name type would add nothing.
variable {Op : Type}

/-- A device: what it executes directly, and how everything else is rewritten. -/
structure Target (Op : Type) where
  /-- Operators the device implements. -/
  prim : Op → Prop
  /-- The operators a rewrite is built from. Empty for something folded away entirely. -/
  rw : Op → List Op

/-- `Reduces T op` means `op` can be built from primitives, in finitely many steps.

Note the second constructor takes a proof for EVERY operator the rewrite mentions. A rewrite
that reaches a primitive along one path and a dead end along another does not qualify, which is
the whole point: the graph has to be emittable in its entirety, not mostly. -/
inductive Reduces (T : Target Op) : Op → Prop where
  | prim {op : Op} : T.prim op → Reduces T op
  | via {op : Op} : (∀ u ∈ T.rw op, Reduces T u) → Reduces T op

/-- A rank witnessing that rewriting makes progress: every operator a rewrite mentions is
strictly simpler than the operator being rewritten.

This is the hypothesis the fixpoint check is really testing. A table that violates it has a
cycle, and a cycle is exactly the shape that makes a naive expander loop rather than fail. -/
def Decreasing (T : Target Op) (rank : Op → Nat) : Prop :=
  ∀ op u, u ∈ T.rw op → rank u < rank op

/-- **The theorem.** With a decreasing rank, every operator reduces -- no case analysis over
the table, no enumeration of 178 operators, and no dependence on which operators are primitive.

So a checker never has to search for a fixpoint to be trusted. It has to exhibit a rank. -/
theorem reduces_of_decreasing (T : Target Op) (rank : Op → Nat)
    (h : Decreasing T rank) (op : Op) : Reduces T op := by
  -- Strong induction on the rank, which is where well-foundedness enters.
  have key : ∀ n op, rank op ≤ n → Reduces T op := by
    intro n
    induction n with
    | zero =>
      intro op hop
      refine Reduces.via ?_
      intro u hu
      -- `rank u < rank op ≤ 0` is impossible, so the rewrite mentions nothing.
      exact absurd (Nat.lt_of_lt_of_le (h op u hu) hop) (Nat.not_lt_zero _)
    | succ n ih =>
      intro op hop
      refine Reduces.via ?_
      intro u hu
      exact ih u (Nat.le_of_lt_succ (Nat.lt_of_lt_of_le (h op u hu) hop))
  exact key (rank op) op (Nat.le_refl _)

/-- An operator whose rewrite is empty reduces immediately. This is the `Identity`, `Constant`
and `Shape` case: folded away, nothing left to emit. -/
theorem reduces_of_rw_nil (T : Target Op) {op : Op} (h : T.rw op = []) : Reduces T op := by
  refine Reduces.via ?_
  intro u hu
  rw [h] at hu
  exact absurd hu (List.not_mem_nil)

/-! ### The hypothesis is load-bearing

A gate that cannot fail certifies the defect, and a theorem whose hypothesis is decorative is
the same fault in another form. So: a target with a cycle, where the conclusion genuinely does
not follow. -/

/-- Two operators that rewrite into each other. Neither is primitive and neither bottoms out. -/
def cyclic : Target Bool where
  prim := fun _ => False
  rw := fun b => [!b]

/-- No rank can be decreasing on `cyclic`: it would need `rank false < rank true` and
`rank true < rank false` at once. So the theorem's hypothesis is exactly what excludes the
looping table, rather than being a convenience. -/
theorem cyclic_has_no_rank : ¬ ∃ rank : Bool → Nat, Decreasing cyclic rank := by
  rintro ⟨rank, h⟩
  have h₁ : rank false < rank true := h true false (by simp [cyclic])
  have h₂ : rank true < rank false := h false true (by simp [cyclic])
  exact absurd (Nat.lt_trans h₁ h₂) (Nat.lt_irrefl _)

/-- And a worked instance in the shape the real table has: a chain of rewrites ending in a
primitive. `Where` mentions `Sub`, `Mul` and `Add`; those are primitive; so `Where` reduces. -/
inductive Toy where
  | add | sub | mul   -- primitive: the device implements these
  | wher              -- rewritten as sub, mul, add
  | gequal            -- rewritten as less, sub -- and `less` is primitive
  | less
  deriving DecidableEq

def toy : Target Toy where
  prim := fun op => op = .add ∨ op = .sub ∨ op = .mul ∨ op = .less
  rw := fun
    | .wher => [.sub, .mul, .add]
    | .gequal => [.less, .sub]
    | _ => []

/-- A rank for the toy target: primitives at 0, rewrites at 1. -/
def toyRank : Toy → Nat
  | .wher | .gequal => 1
  | _ => 0

theorem toy_decreasing : Decreasing toy toyRank := by
  intro op u hu
  cases op with
  | add => simp [toy] at hu
  | sub => simp [toy] at hu
  | mul => simp [toy] at hu
  | less => simp [toy] at hu
  | wher =>
      simp only [toy, List.mem_cons, List.not_mem_nil, or_false] at hu
      rcases hu with rfl | rfl | rfl <;> decide
  | gequal =>
      simp only [toy, List.mem_cons, List.not_mem_nil, or_false] at hu
      rcases hu with rfl | rfl <;> decide

/-- Every operator in the toy target reduces, by the theorem rather than by inspection. -/
theorem toy_reduces (op : Toy) : Reduces toy op :=
  reduces_of_decreasing toy toyRank toy_decreasing op

end Reduction
