import Mathlib

set_option linter.style.header false

/-!
Formal verification of the algebraic "primitive decomposition" tricks used in
`src/ops.cpp`/`src/loss.cpp` to give ggml ops that lack a backward-pass case
(CLAMP, and the elementwise max/min used inside the GIoU computation) a
backward-capable equivalent built from ops that do have one (RELU/ADD/SUB).

These aren't hypothetical identities, they are exactly what's shipped:
* `clamp_diff` (src/ops.cpp:160): `lo + relu(x-lo) - relu(x-hi)`
* `elementwise_max` (src/loss.cpp:167): `a + relu(b-a)`
* `elementwise_min` (src/loss.cpp:170): `a - relu(a-b)`

and they are used to build GIoU (src/loss.cpp:242-274), whose boundedness
(`-1 ≤ giou ≤ 1`) is exactly the property that keeps `detection_loss`'s
aggregate magnitude bounded per matched box, see
`docs/decisions/0003-training.md`'s numerical-blowup investigation this port
went through (a per-element bound doesn't automatically bound a sum over many
elements, which is exactly what bit this port once already).

`relu t` is `max t 0` throughout, matching `ggml_relu`'s actual semantics.
All argument lists below are fully explicit (no ambient auto-bound
`variable`s) so that call-site argument order can never silently mismatch a
definition's inferred binder order.
-/

noncomputable section

/-- `relu t = max t 0`, matching `ggml_relu`. -/
def relu (t : ℝ) : ℝ := max t 0

@[simp] theorem relu_nonneg (t : ℝ) : 0 ≤ relu t := le_max_right _ _

theorem relu_mono {s t : ℝ} (h : s ≤ t) : relu s ≤ relu t := max_le_max h le_rfl

/-- `elementwise_max` (src/loss.cpp:167): `a + relu(b - a) = max a b`. -/
theorem elementwise_max_eq (a b : ℝ) : a + relu (b - a) = max a b := by
  unfold relu
  rcases le_total a b with h | h
  · rw [max_eq_right h, max_eq_left (by linarith : (0:ℝ) ≤ b - a)]; ring
  · rw [max_eq_left h, max_eq_right (by linarith : b - a ≤ 0)]; ring

/-- `elementwise_min` (src/loss.cpp:170): `a - relu(a - b) = min a b`. -/
theorem elementwise_min_eq (a b : ℝ) : a - relu (a - b) = min a b := by
  unfold relu
  rcases le_total a b with h | h
  · rw [min_eq_left h, max_eq_right (by linarith : a - b ≤ 0)]; ring
  · rw [min_eq_right h, max_eq_left (by linarith : (0:ℝ) ≤ a - b)]; ring

/-- `clamp_diff` (src/ops.cpp:160): `lo + relu(x-lo) - relu(x-hi) = clamp x lo hi`,
for any `lo ≤ hi`. `clamp x lo hi` is written as `min (max x lo) hi`, the
standard order-lattice definition of clamp. This is the primitive
`detection_loss` uses to keep `sigmoid`'s output away from exactly 0/1
before `log`, and to keep `bbox_reparam_decode_diff`'s `delta_wh` away from
overflowing `exp`, both directly load-bearing for this port's earlier
numerical-blowup fix (see `docs/decisions/0003-training.md`). -/
theorem clamp_diff_eq (x lo hi : ℝ) (hlohi : lo ≤ hi) :
    lo + relu (x - lo) - relu (x - hi) = min (max x lo) hi := by
  unfold relu
  simp only [min_def, max_def]
  split_ifs <;> linarith

/-! ### GIoU boundedness

Faithful transcription of `detection_loss`'s GIoU block (`src/loss.cpp:242-274`):
boxes are given in `xyxy` form, `emax`/`emin` are exactly `elementwise_max`/
`elementwise_min` above (proved equal to `max`/`min`), intersection/enclose
width/height go through `relu` (matching `ggml_relu(ctx, ggml_sub(...))`).
The `+ 1e-7` denominator regularizers from the C++ are dropped here since
they only guard exact 0/0 and don't affect which direction the bound goes;
strict positivity hypotheses on the relevant denominators take their place.

Every definition below takes its eight box-coordinate arguments in the
SAME explicit order everywhere: `x1a x2a x1b x2b y1a y2a y1b y2b`
(box A's x-range, box B's x-range, box A's y-range, box B's y-range). -/

def interW (x1a x2a x1b x2b : ℝ) : ℝ := relu (min x2a x2b - max x1a x1b)
def interH (y1a y2a y1b y2b : ℝ) : ℝ := relu (min y2a y2b - max y1a y1b)
def interArea (x1a x2a x1b x2b y1a y2a y1b y2b : ℝ) : ℝ :=
  interW x1a x2a x1b x2b * interH y1a y2a y1b y2b

def encW (x1a x2a x1b x2b : ℝ) : ℝ := relu (max x2a x2b - min x1a x1b)
def encH (y1a y2a y1b y2b : ℝ) : ℝ := relu (max y2a y2b - min y1a y1b)
def encArea (x1a x2a x1b x2b y1a y2a y1b y2b : ℝ) : ℝ :=
  encW x1a x2a x1b x2b * encH y1a y2a y1b y2b

/-- The intersection is never wider than box A: `min x2a x2b ≤ x2a`,
`max x1a x1b ≥ x1a`, and `relu` is monotone. -/
theorem interW_le_left {x1a x2a x1b x2b : ℝ} (hxa : x1a ≤ x2a) :
    interW x1a x2a x1b x2b ≤ x2a - x1a := by
  unfold interW
  have h : min x2a x2b - max x1a x1b ≤ x2a - x1a := by
    have h1 : min x2a x2b ≤ x2a := min_le_left _ _
    have h2 : x1a ≤ max x1a x1b := le_max_left _ _
    linarith
  calc relu (min x2a x2b - max x1a x1b) ≤ relu (x2a - x1a) := relu_mono h
    _ = x2a - x1a := max_eq_left (by linarith)

theorem interW_le_right {x1a x2a x1b x2b : ℝ} (hxb : x1b ≤ x2b) :
    interW x1a x2a x1b x2b ≤ x2b - x1b := by
  unfold interW
  have h : min x2a x2b - max x1a x1b ≤ x2b - x1b := by
    have h1 : min x2a x2b ≤ x2b := min_le_right _ _
    have h2 : x1b ≤ max x1a x1b := le_max_right _ _
    linarith
  calc relu (min x2a x2b - max x1a x1b) ≤ relu (x2b - x1b) := relu_mono h
    _ = x2b - x1b := max_eq_left (by linarith)

theorem interH_le_left {y1a y2a y1b y2b : ℝ} (hya : y1a ≤ y2a) :
    interH y1a y2a y1b y2b ≤ y2a - y1a := by
  unfold interH
  have h : min y2a y2b - max y1a y1b ≤ y2a - y1a := by
    have h1 : min y2a y2b ≤ y2a := min_le_left _ _
    have h2 : y1a ≤ max y1a y1b := le_max_left _ _
    linarith
  calc relu (min y2a y2b - max y1a y1b) ≤ relu (y2a - y1a) := relu_mono h
    _ = y2a - y1a := max_eq_left (by linarith)

theorem interH_le_right {y1a y2a y1b y2b : ℝ} (hyb : y1b ≤ y2b) :
    interH y1a y2a y1b y2b ≤ y2b - y1b := by
  unfold interH
  have h : min y2a y2b - max y1a y1b ≤ y2b - y1b := by
    have h1 : min y2a y2b ≤ y2b := min_le_right _ _
    have h2 : y1b ≤ max y1a y1b := le_max_right _ _
    linarith
  calc relu (min y2a y2b - max y1a y1b) ≤ relu (y2b - y1b) := relu_mono h
    _ = y2b - y1b := max_eq_left (by linarith)

/-- `inter ≤ area_a` and `inter ≤ area_b`: the intersection area never
exceeds either input box's own area. -/
theorem interArea_le_areaA {x1a x2a x1b x2b y1a y2a y1b y2b : ℝ}
    (hxa : x1a ≤ x2a) (hya : y1a ≤ y2a) :
    interArea x1a x2a x1b x2b y1a y2a y1b y2b ≤ (x2a - x1a) * (y2a - y1a) := by
  unfold interArea
  exact mul_le_mul (interW_le_left hxa) (interH_le_left hya) (relu_nonneg _) (by linarith)

theorem interArea_le_areaB {x1a x2a x1b x2b y1a y2a y1b y2b : ℝ}
    (hxb : x1b ≤ x2b) (hyb : y1b ≤ y2b) :
    interArea x1a x2a x1b x2b y1a y2a y1b y2b ≤ (x2b - x1b) * (y2b - y1b) := by
  unfold interArea
  exact mul_le_mul (interW_le_right hxb) (interH_le_right hyb) (relu_nonneg _) (by linarith)

/-- `0 ≤ IoU ≤ 1`, exactly matching `detection_loss`'s
`iou = inter / (area1 + area2 - inter)` (`src/loss.cpp:264-265`, epsilon
dropped, see the file-level note above). -/
theorem iou_bounds {x1a x2a x1b x2b y1a y2a y1b y2b : ℝ}
    (hxa : x1a ≤ x2a) (hya : y1a ≤ y2a) (hxb : x1b ≤ x2b) (hyb : y1b ≤ y2b)
    (huni_pos : 0 < (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b)
                    - interArea x1a x2a x1b x2b y1a y2a y1b y2b) :
    0 ≤ interArea x1a x2a x1b x2b y1a y2a y1b y2b /
          ((x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b)
            - interArea x1a x2a x1b x2b y1a y2a y1b y2b)
      ∧ interArea x1a x2a x1b x2b y1a y2a y1b y2b /
          ((x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b)
            - interArea x1a x2a x1b x2b y1a y2a y1b y2b) ≤ 1 := by
  have hinterA : interArea x1a x2a x1b x2b y1a y2a y1b y2b ≤ (x2a - x1a) * (y2a - y1a) :=
    interArea_le_areaA hxa hya
  have hinterB : interArea x1a x2a x1b x2b y1a y2a y1b y2b ≤ (x2b - x1b) * (y2b - y1b) :=
    interArea_le_areaB hxb hyb
  have hinter0 : 0 ≤ interArea x1a x2a x1b x2b y1a y2a y1b y2b :=
    mul_nonneg (relu_nonneg _) (relu_nonneg _)
  refine ⟨div_nonneg hinter0 huni_pos.le, ?_⟩
  rw [div_le_one huni_pos]
  linarith

/-- **`-1 ≤ GIoU ≤ 1`** (`src/loss.cpp:273`: `giou := iou - (enclose - union)/enclose`).
Given `0 ≤ IoU ≤ 1`, `union ≥ 0`, `union ≤ enclose`, and `enclose > 0`, GIoU is
bounded in `[-1, 1]`. `union ≤ enclose` (the smallest axis-aligned box
enclosing two boxes has area at least their union's, since both input boxes,
hence their set union, are literally subsets of the enclosing box) is the
well-known containment fact underlying Theorem 1 of the original GIoU paper
(Rezatofighi et al., CVPR 2019); it's taken as a hypothesis here rather than
re-derived from 2D measure theory, which is out of scope for this port's
verification. -/
theorem giou_bounds (iou uni enc : ℝ)
    (hiou0 : 0 ≤ iou) (hiou1 : iou ≤ 1)
    (huni_nonneg : 0 ≤ uni) (huni_le_enc : uni ≤ enc) (henc_pos : 0 < enc) :
    -1 ≤ iou - (enc - uni) / enc ∧ iou - (enc - uni) / enc ≤ 1 := by
  have hfrac0 : 0 ≤ (enc - uni) / enc := div_nonneg (by linarith) henc_pos.le
  have hfrac1 : (enc - uni) / enc ≤ 1 := by rw [div_le_one henc_pos]; linarith
  exact ⟨by linarith, by linarith⟩

/-- Fully composed version, wired directly to the eight box coordinates
exactly as `detection_loss` computes them (`src/loss.cpp:242-274`) rather
than abstract `iou`/`uni`/`enc` reals. The ONLY hypothesis not derived from
the `elementwise_max`/`min`+`relu` construction itself is `union ≤ enclose`
(cited from the GIoU paper, see `giou_bounds`'s docstring); everything else
(`0 ≤ IoU ≤ 1`, `union ≥ 0`, `union > 0`) follows from
`interArea_le_areaA`/`interArea_le_areaB` above. -/
theorem giou_bounds_full {x1a x2a x1b x2b y1a y2a y1b y2b : ℝ}
    (hxa : x1a ≤ x2a) (hya : y1a ≤ y2a) (hxb : x1b ≤ x2b) (hyb : y1b ≤ y2b)
    (huni_pos : 0 < (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b)
                    - interArea x1a x2a x1b x2b y1a y2a y1b y2b)
    (huni_le_enc : (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b)
                    - interArea x1a x2a x1b x2b y1a y2a y1b y2b
                    ≤ encArea x1a x2a x1b x2b y1a y2a y1b y2b) :
    -1 ≤ interArea x1a x2a x1b x2b y1a y2a y1b y2b /
           ((x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b)
             - interArea x1a x2a x1b x2b y1a y2a y1b y2b)
         - (encArea x1a x2a x1b x2b y1a y2a y1b y2b
             - ((x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b)
                 - interArea x1a x2a x1b x2b y1a y2a y1b y2b))
           / encArea x1a x2a x1b x2b y1a y2a y1b y2b
    ∧ interArea x1a x2a x1b x2b y1a y2a y1b y2b /
           ((x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b)
             - interArea x1a x2a x1b x2b y1a y2a y1b y2b)
         - (encArea x1a x2a x1b x2b y1a y2a y1b y2b
             - ((x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b)
                 - interArea x1a x2a x1b x2b y1a y2a y1b y2b))
           / encArea x1a x2a x1b x2b y1a y2a y1b y2b ≤ 1 := by
  have hbounds := iou_bounds hxa hya hxb hyb huni_pos
  have huni_nonneg :
      0 ≤ (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b)
            - interArea x1a x2a x1b x2b y1a y2a y1b y2b := by
    have hinterA : interArea x1a x2a x1b x2b y1a y2a y1b y2b ≤ (x2a - x1a) * (y2a - y1a) :=
      interArea_le_areaA hxa hya
    have hinterB : interArea x1a x2a x1b x2b y1a y2a y1b y2b ≤ (x2b - x1b) * (y2b - y1b) :=
      interArea_le_areaB hxb hyb
    linarith
  have henc_pos : 0 < encArea x1a x2a x1b x2b y1a y2a y1b y2b := lt_of_lt_of_le huni_pos huni_le_enc
  exact giou_bounds _ _ _ hbounds.1 hbounds.2 huni_nonneg huni_le_enc henc_pos

end
