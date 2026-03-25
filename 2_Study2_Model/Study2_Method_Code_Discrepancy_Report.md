# Study 2 Method vs Code Discrepancy Report

## Context (for future turns)
- This report was created after a request to **deeply compare** the Study 2 Method text (including formulas) in `Manuscript/DesignSpaceSPE.Rmd` against the actual simulation implementation in `Study2/S2_gen_data_optimized.ipynb` (forget the legacy notebook `Study2/S2_gen_data.ipynb`).
- Prior conversation context:
  - The notebook has been iteratively updated (including `a` variability and plotting updates for `a`/`v`).
  - The user requested **analysis only** at this stage (no manuscript editing yet).
- Goal of this report:
  1. Identify discrepancies between manuscript Method (Study 2) and notebook code.
  2. Identify incompleteness in the manuscript Method.
  3. Provide scientifically grounded suggestions.
  4. Provide useful references verified to exist in the literature.

---

## Scope and source files
- Manuscript source: `Manuscript/DesignSpaceSPE.Rmd` (Study 2 Method section around lines 271-386).
- Simulation source: `Study2/S2_gen_data_optimized.ipynb` (model + generation functions).

## Quick start (next session)
- Status snapshot (Section 1): `D1, D2, D3, D4, D5, D6, D8, D9` solved; `D7` partially solved.
- Notebook split status (D9):
  - `Study2/S2_gen_data_optimized.ipynb` now keeps random-space generation only.
  - `Study3/S3_gen_data.ipynb` now keeps fixed-design generation only (large-dataset block removed).
- Highest-priority remaining task: finish `D7` with an explicit operational mapping (formula/table) from simulator `response` to correctness/ACC coding.
- Then move to Section 2 (`I1-I7`), starting with `I3` (ACC mapping), `I1` (parameter table), and `I2` (reproducibility protocol).
- Files to touch next:
  - `Manuscript/DesignSpaceSPE.Rmd` (add exact ACC mapping rule and, later, Study 3 method text for fixed-design generator provenance).
  - `Study3/S3_fan.Rmd` and/or `Study3/S3_fan_refactored.Rmd` (add Study 3 method note for the moved fixed-design generator).
- Fast verification checklist before closing next session:
  - Confirm Section 1 statuses in this report match current code/text.
  - Confirm no fixed-design symbols remain in `Study2/S2_gen_data_optimized.ipynb`.
  - Confirm `Study3/S3_gen_data.ipynb` has fixed-design generation and no large random dataset block.

---

## 1) Discrepancies between Study 2 Method text/formulas and notebook implementation

### ~~D1. Logistic sign direction is opposite between written equations and implemented equations (critical)~~
**Method text/equations (increasing claim):**
- The prose explicitly says increasing-and-saturating relationships for *P*, *T*, and *M* (e.g., "as *P* increases ... drift rate *v* continuously increases" and similarly for *T*, *M*) in `Manuscript/DesignSpaceSPE.Rmd:301`, `Manuscript/DesignSpaceSPE.Rmd:313`, and `Manuscript/DesignSpaceSPE.Rmd:327`.
- But the formulas are written as:
  - `v_P = 1 / (1 + e^{k_P(P - P_0)})` in `Manuscript/DesignSpaceSPE.Rmd:303`
  - `v_T = 1 / (1 + e^{k_T(T - T_0)})` in `Manuscript/DesignSpaceSPE.Rmd:315`
  - `a = 1 / (1 + e^{k_M(M - M_0)})` in `Manuscript/DesignSpaceSPE.Rmd:329`

With positive slopes, those are decreasing sigmoids.

**Notebook code (increasing implementation):**
- `return 1 / (1 + np.exp(-k * (P - P1)))` in `Study2/S2_gen_data_optimized.ipynb:232`.
- `v_T = 1 / (1 + np.exp(-k_T * (T - T_0)))` in `Study2/S2_gen_data_optimized.ipynb:240`.
- `a_0 = 1 / (1 + np.exp(-k * (M - M_0))) * 3` in `Study2/S2_gen_data_optimized.ipynb:258`.

**Impact:** The manuscript formulas currently encode the opposite monotonic direction from the implemented generator.

> Solution: Now fixed, already chanaged the text in Study2 Methods section.
>
> Verification (current): **Solved**. Study 2 method equations now use consistent increasing-sigmoid signs for `v_P`, `v_T`, and `a` (`Manuscript/DesignSpaceSPE.Rmd:303`, `Manuscript/DesignSpaceSPE.Rmd:315`, `Manuscript/DesignSpaceSPE.Rmd:329`, `Manuscript/DesignSpaceSPE.Rmd:339`).

---

### ~~D2. `v_P` parameterization is not the same as the written formula (`P0` vs `P1`, ~~plus slope argument reversal~~)~~
**Method text/equations:**
- `v_P` is written as a function centered at `P_0` (`Manuscript/DesignSpaceSPE.Rmd:303`) with `k_P` separately defined (`Manuscript/DesignSpaceSPE.Rmd:305`).

**Notebook code:**
- `k_P` uses `P0=32` in `k_P_func` (`Study2/S2_gen_data_optimized.ipynb:224-226`).
- `v_P` uses a second center parameter `P1=4` in `v_P_func` (`Study2/S2_gen_data_optimized.ipynb:229-233`).
- In `compute_v`, the call is `v_P_func(..., P1=4, k_min=0.1, k_max=0.05, ...)` (`Study2/S2_gen_data_optimized.ipynb:241`), i.e., the lower/upper bounds are passed in reversed order relative to the default `k_min < k_max` pattern.

**Impact:** A reader cannot reproduce notebook behavior from the manuscript equation alone, because the implemented `v_P` center and slope parameterization differ from the text.

> Solution: fixed the lower/upper bounds, which might be caused by typos. to-do list: should add more details in manuscript text.
>
> Verification (current): **Solved (Option A)**. Manuscript now specifies `P_1` in `v_P` and `P_0` in `k_P` (`Manuscript/DesignSpaceSPE.Rmd:303`, `Manuscript/DesignSpaceSPE.Rmd:305`), consistent with code structure in `Study2/S2_gen_data_optimized.ipynb`.
---

### ~~D3. Complete-model equations omit implemented scaling constants and piecewise boundary modulation~~
**Method text/equations:**
- `v = v_P * v_T * (1 + alpha)` in `Manuscript/DesignSpaceSPE.Rmd:335`.
- `a = logistic(M)` in `Manuscript/DesignSpaceSPE.Rmd:337`.

**Notebook code:**
- Drift is globally scaled: `v_0 = v_T * v_P * 3` in `Study2/S2_gen_data_optimized.ipynb:242`.
- Boundary is globally scaled: `a_0 = logistic(M) * 3` in `Study2/S2_gen_data_optimized.ipynb:258`.
- Boundary has a piecewise multiplier not present in the method text:
  - `if M > 600: a_1 = a_0 * (1 + BETA1)` (`Study2/S2_gen_data_optimized.ipynb:260-261`)
  - `else: a_1 = a_0 * (1 + BETA2)` (`Study2/S2_gen_data_optimized.ipynb:262-263`)

**Impact:** The written complete model is under-specified relative to the code that generates the reported synthetic data.

> Solution:  to-do, should add more details in manuscript text. The justification of a global scale factor 3 is missing, we may need to reconsider it.
>
> Verification (current): **Solved**. The complete-model block now includes retained scaling (`v_0 = v_P \cdot v_T \cdot 3`, `a_0 = logistic(M) \cdot 3`) plus piecewise boundary modulation with `\beta_1/\beta_2` (`Manuscript/DesignSpaceSPE.Rmd:335-346`), matching implementation.

---

### ~~D4. Boundary variability distribution differs from text (and differs across notebooks)~~
**Method text:**
- "Decision threshold *a* varies at the subject level, randomly drawn from normal distribution `N(a_0, 1)`" in `Manuscript/DesignSpaceSPE.Rmd:349`.

**Notebook code (optimized notebook):**
- `a = np.random.normal(a_base, a_base * A_CV)` in `Study2/S2_gen_data_optimized.ipynb:319` and `Study2/S2_gen_data_optimized.ipynb:431`.
- `A_CV = 0.15` is defined in `Study2/S2_gen_data_optimized.ipynb:193`.

**~~Related notebook difference (legacy notebook):~~**
- ~~In `Study2/S2_gen_data.ipynb` (cell 2), boundary noise is coded as `a = np.random.normal(a_1, 0.5)` (cell line 69), not `N(a_0, 1)` and not CV-scaled.~~

**Impact:** The manuscript variance statement does not match the optimized implementation, and notebook versions also differ in their `a` noise model.

> Solution: should only consider improved .ipynb. todo: should find a optimal a_CV for the between subject level variability.
>
> Verification (current): **Solved**. Manuscript now states `N(a_0, a_0 \cdot A_{CV})` with `A_{CV}=0.15` (`Manuscript/DesignSpaceSPE.Rmd:360`), aligned with code (`Study2/S2_gen_data_optimized.ipynb`).
---

### ~~D5. Uniform sampling statement is not equivalent to implemented integer sampling bounds~~
**Method text:**
- `P ~ Uniform(0, 150)`, `T ~ Uniform(10, 600)`, `W ~ Uniform(200, 1500)` in `Manuscript/DesignSpaceSPE.Rmd:345-347`.

**Notebook code:**
- `np.random.randint(10, 600)`, `np.random.randint(0, 150)`, `np.random.randint(200, 1500)` in `Study2/S2_gen_data_optimized.ipynb:424-426` and vectorized equivalents in `Study2/S2_gen_data_optimized.ipynb:501-503`.

`np.random.randint(low, high)` excludes `high`, so the effective support is `[0,149]`, `[10,599]`, `[200,1499]`.

**Impact:** Small but real mismatch in design-space support and replicability wording.

> Solution: update text to match the code, which is more realistic.
>
> Verification (current): **Solved**. Manuscript now uses discrete integer supports (`P \in {0..149}`, `T \in {10..599}`, `W \in {200..1499}`) in `Manuscript/DesignSpaceSPE.Rmd:356-358`, consistent with `np.random.randint` in `Study2/S2_gen_data_optimized.ipynb`.

---

### ~~D6. Maximum-time stopping rule in text is simplified relative to implementation~~
**Method text:**
- Stopping condition says completion must occur within maximum time `T + W` (`Manuscript/DesignSpaceSPE.Rmd:355`).

**Notebook code:**
- `max_time = (W + T - T0) * 0.001 + 0.001` in `Study2/S2_gen_data_optimized.ipynb:327` and `Study2/S2_gen_data_optimized.ipynb:439`.
- Trial acceptance requires `response > 0 and decision_time < max_time` in `Study2/S2_gen_data_optimized.ipynb:344` and `Study2/S2_gen_data_optimized.ipynb:453`.

**Impact:** The true criterion includes non-decision-time subtraction and a 1 ms offset, so it is not exactly equivalent to `T + W` as written.

> Solution: the code is more realistic, we can update the text.
>
> Verification (current): **Solved**. Manuscript now mirrors implementation with explicit `t_{max}` formula and validity criterion (`response > 0` and `decision_time < t_{max}`) in `Manuscript/DesignSpaceSPE.Rmd:368-370`.

---

### D7. Outcome variable semantics (ACC vs bound-hit `response`) are not explicitly bridged
**Method text:**
- Data-analysis text discusses ACC effect sizes (`Manuscript/DesignSpaceSPE.Rmd:359-365`).

**Notebook code:**
- Generator writes `response` (bound reached) and does not create `Match`, `CorrectKey`, or `Correct` at generation stage in `Study2/S2_gen_data_optimized.ipynb:345-356` and `Study2/S2_gen_data_optimized.ipynb:454-462`.

**Impact:** Without an explicit mapping from bound-hit to task correctness, ACC interpretation is under-documented.

> Solution: This is a great question, pointed out a trick not mentioned in text. The simulation actually only simulated half of the trils, the matching trials, because previous studies have only found SPE in the matching trials. We should add this information in both the text and the notebook.
>
> Verification (current): **Partially solved**. Manuscript now states that simulator output is boundary-hit `response` and that ACC interpretation uses a post-simulation mapping rule under match-trial scope (`Manuscript/DesignSpaceSPE.Rmd:372`).
>
> Recommendation: Add the exact operational mapping rule (formula/table) from `response` to correctness coding used in ACC computation, so it is fully reproducible.

---

### ~~D8. Notation typo changes interpretation in the boundary section~~
**Method text:**
- The boundary equation section says `$k_M$ is the slope parameter describing the sensitivity of drift rate *v* to M` (`Manuscript/DesignSpaceSPE.Rmd:327`), but the equation on that line models `a(M)` (`Manuscript/DesignSpaceSPE.Rmd:329`).

**Notebook code:**
- `compute_a(M)` clearly uses `M` to compute boundary, not drift (`Study2/S2_gen_data_optimized.ipynb:253-265`).

**Impact:** Conceptual labeling inconsistency can mislead readers about whether `M` modulates `v` or `a`.

> Solution: typo, fixed already.
>
> Verification (current): **Solved**. Manuscript now correctly describes `$k_M$` as sensitivity of threshold `a` to `M` (`Manuscript/DesignSpaceSPE.Rmd:327`).
---

### ~~D9. Two Study-2 generation regimes are conflated in prose unless explicitly separated~~
**Method text:**
- Model specification states large random-space generation: 300000 subjects x 260 trials (`Manuscript/DesignSpaceSPE.Rmd:343`).

**Notebook code (now split by study):**
- Study 2 notebook (`Study2/S2_gen_data_optimized.ipynb`) now keeps random-space generation only, including the large random dataset path.
- Study 3 notebook (`Study3/S3_gen_data.ipynb`) now hosts fixed design-point generation (`EXPERIMENT_CONDITIONS` + `generate_condition_files(...)`).

**Impact:** If manuscript text does not separate these two outputs, readers can misinterpret what data source underlies each downstream analysis.

> Solution: Fixed design-point generator are generating data that match study 3's empirical designs. We should add a note explicitly exclude this part of the code from study 2.
>
> Verification (current): **Solved by code-level separation**. The workflow split removes cross-study mixing in implementation. Study 2 method text intentionally remains unchanged at this stage per current plan.

---

## 2) Incompleteness in Study 2 Method (manuscript)

### I1. Missing explicit parameter table for “model as implemented”
Not all constants are listed in one place with values and units (`ALPHA1`, `ALPHA2`, `BETA1`, `BETA2`, `GAMMA`, `T0`, `DELTA_T`, `A_CV`, offsets like `P1`, `P0`, `T_0`, `M_0`, scaling `*3`).

### I2. Missing exact randomization and reproducibility protocol
Need explicit seeding strategy and whether sampling is independent with replacement, plus endpoint conventions.

### I3. Missing explicit mapping from latent DDM choice to behavioral ACC
A full generative definition of match/mismatch, correct key, and accuracy coding is necessary.

### I4. Missing explicit statement on invalid-trial handling
Because non-terminated trials are dropped, effective per-condition trial counts vary. This should be stated as it affects effect-size estimates.

### I5. Missing computational implementation details for optimized pipeline
Method should mention batch vectorization and parallelization if these outputs are used for final reported model summaries.

### I6. Placeholder citations remain
Study 2 has “citation needed” placeholders (e.g., effect-size analysis and GAM details). These must be resolved before submission.

### I7. Unit harmonization is incomplete
Method mixes ms and seconds in prose without a single canonical convention section.

> Solution: we should add these details to the text. The issues I7 should be uified in both text and the notebook.

### Status check for Section 2 (current)
- **Solved:** none fully solved yet.
- **Unsolved:** `I1-I7` remain open based on current `Manuscript/DesignSpaceSPE.Rmd` and `Study2/S2_gen_data_optimized.ipynb`.

Recommendations by item:
- `I1` Parameter table: add one compact table with symbol, code variable, value, unit, and where applied (equation/function).
- `I2` Randomization protocol: document seed policy, endpoint convention (`randint` upper-exclusive), and whether generation is independent by subject.
- `I3` ACC mapping: define exactly how latent bound outcomes map to behavioral correctness, including match-only scope if used.
- `I4` Invalid-trial handling: state exclusion criterion and report attrition rate summary.
- `I5` Implementation details: include optimized generation path (batch vectorization + joblib parallel) and when it is used for reported outputs.
- `I6` Citations: resolve all "citation needed" markers in Study 2 text.
- `I7` Units: choose one canonical unit convention (recommended: seconds in code-facing formulas; ms only for UI/task-setting prose with explicit conversion).

---

## 3) Scientific suggestions (non-editing recommendations)

1. **Add a “Model-as-Implemented” subsection**
   - Include final equations exactly matching code, with parameter values and units.

2. **Separate two simulation products clearly**
   - (a) Full random-space synthetic dataset.
   - (b) Fixed design-point synthetic datasets used for empirical comparison.

3. **Formalize ACC generation**
   - Explicitly model trial type (`match/mismatch`), response rule, and `Correct` coding from latent choice.

4. **Report trial attrition and effective n**
   - Provide valid-trial rates by condition and discuss potential bias.

5. **Add parameter sensitivity analyses**
   - At least for `A_CV`, slope constants, and threshold modulation terms to assess robustness of boundary claims.

6. **Include recovery/identifiability checks**
   - Especially for distinguishing drift (`v`) changes from boundary (`a`) changes under varying `W` and `T`.

7. **Provide reproducibility metadata**
   - Exact runtime environment and script/notebook version used for reported tables/figures.

8. **Correct sign and notation in equations**
   - Ensure formula monotonicity aligns with verbal hypotheses and implementation.
  
> Solutions: we will renew this part after solving the issues in 1) and 2).

Status check for Section 3 (current):
- **Unsolved/pending by dependency:** keep this section pending until Section 1/2 updates are finished.
- Priority recommendation: implement in this order -> `#8` (equation sign consistency), `#1` (model-as-implemented block), `#3` (ACC formalization), `#4` (attrition reporting), then reproducibility/sensitivity items.

---

## 4) Verified useful references (existence checked)

Below are references relevant to the Study 2 Method claims and implementation. These were verified via PubMed / journal pages.

1. **Ratcliff, R., & McKoon, G. (2008).**
   *The diffusion decision model: theory and data for two-choice decision tasks.*
   Neural Computation, 20(4), 873-922.
   DOI: https://doi.org/10.1162/neco.2008.12-06-420
   PubMed: https://pubmed.ncbi.nlm.nih.gov/18085991/

2. **Heitz, R. P. (2014).**
   *The speed-accuracy tradeoff: history, physiology, methodology, and behavior.*
   Frontiers in Neuroscience, 8:150.
   DOI: https://doi.org/10.3389/fnins.2014.00150
   PubMed: https://pubmed.ncbi.nlm.nih.gov/24966810/

3. **Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D. (2006).**
   *The physics of optimal decision making: A formal analysis of models of performance in two-alternative forced-choice tasks.*
   Psychological Review, 113(4), 700-765.
   DOI: https://doi.org/10.1037/0033-295X.113.4.700

4. **Bogacz, R., Hu, P. T., Holmes, P. J., & Cohen, J. D. (2010).**
   *Do humans produce the speed-accuracy trade-off that maximizes reward rate?*
   Quarterly Journal of Experimental Psychology, 63(5), 863-891.
   DOI: https://doi.org/10.1080/17470210903091643
   PubMed: https://pubmed.ncbi.nlm.nih.gov/19746300/

5. **Ratcliff, R., & Smith, P. L. (2004).**
   *A comparison of sequential sampling models for two-choice reaction time.*
   Psychological Review, 111(2), 333-367.
   DOI: https://doi.org/10.1037/0033-295X.111.2.333
   PubMed: https://pubmed.ncbi.nlm.nih.gov/15065913/

6. **Voss, A., Voss, J., & Lerche, V. (2015).**
   *Assessing cognitive processes with diffusion model analyses: a tutorial based on fast-dm-30.*
   Frontiers in Psychology, 6:336.
   DOI: https://doi.org/10.3389/fpsyg.2015.00336
   Article: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00336/full

7. **Sui, J., He, X., & Humphreys, G. W. (2012).**
   *Perceptual effects of social salience: evidence from self-prioritization effects on perceptual matching.*
   Journal of Experimental Psychology: Human Perception and Performance, 38(5), 1105-1117.
   DOI: https://doi.org/10.1037/a0029792
   PubMed: https://pubmed.ncbi.nlm.nih.gov/22963229/

8. **Turk, D. J., Cunningham, S. J., & Macrae, C. N. (2008).**
   *Self-memory biases in explicit and incidental encoding of trait adjectives.*
   Consciousness and Cognition, 17(3), 1040-1045.
   DOI: https://doi.org/10.1016/j.concog.2008.02.004
   PubMed: https://pubmed.ncbi.nlm.nih.gov/18395467/

9. **Murphy, P. R., Robertson, I. H., Harty, S., & O'Connell, R. G. (2015).**
   *Neural evidence accumulation persists after choice to inform metacognitive judgments.*
   eLife, 4:e11946.
   DOI: https://doi.org/10.7554/eLife.11946
   PubMed: https://pubmed.ncbi.nlm.nih.gov/26687008/

> Note: The manuscript cites `Zylberberg2014` for post-stimulus accumulation. I did not confirm a 2014 paper with that exact claim matching the current citation context; the closest verified direct statement found is Murphy et al. (2015, eLife).

---

## Suggested next step (when editing is allowed)
- Produce a manuscript-ready replacement for Study 2 Method equations + model specification that is fully code-aligned, then run a consistency pass across Study 3/4 sections that reference Study 2 predictions.
