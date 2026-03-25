# AGENTS_GEN_MODELS.md

## Session Starter Prompt (Reuse In New Sessions)

Copy and paste this at the beginning of a new session:

```text
Read `Study2/AGENTS_GEN_MODELS.md` fully, then continue exactly from the current tracker state.

Rules:
1) Follow the phase workflow in this file.
2) Update the tracker/status fields when work is done.
3) Keep outputs reproducible and save figures/scripts with stable names.

Current continuation target:
- Primary target: `gen_rand_data` (HDDM)
- Start from: Phase 1 (source mapping), unless tracker already advanced.

Phase 1 deliverables (no code edits yet):
1) Exact implementation path (wrapper -> helpers -> low-level kernels).
2) Call graph from entry function to sampling core.
3) Parameter semantics table (defaults, transforms, scaling, constraints, RNG behavior).
4) File/URL reference list.
5) Draft parity test plan (metrics, quantiles, parameter sets, tolerances).

After Phase 1 summary is approved, proceed to:
- Phase 2: create skeleton files
  - `Study2/gen_data_hddm.py`
  - `Study2/verify_hddm_gen.py`

Visual standard for Phase 5:
- One combined panel figure
  - Row 1: histogram + density overlays
  - Row 2: cumulative quantile plots (q10, q30, q50, q70, q90)

Before ending the session:
1) Update `Study2/AGENTS_GEN_MODELS.md` (Session Log + tracker + Files Generated).
2) Report open gaps and exact next step.
```

This file tracks goals, progress, workflow, and generated outputs for reverse engineering data generation algorithms in mainstream DDM software.

## Goal

Reverse engineer and reproduce data-generating algorithms in major DDM tools.

## Targets

| target | source | status | notes |
|---|---|---|---|
| `rdiffusion` | [`rtdists::diffusion.R`](https://github.com/rtdists/rtdists/blob/master/R/diffusion.R) | mostly done | fastdm-style Python rewrite implemented and compared |
| `gen_rand_data` | [`hddm/generate.py`](https://github.com/hddm-devs/hddm/blob/master/hddm/generate.py) | not started | next target |
| more targets | TBD | not started | add as needed |

## Current Status Tracker

| target | phase | task | owner | status | date | notes |
|---|---|---|---|---|---|---|
| `rdiffusion` | Phase 1-3 | reverse engineer + python port (`fastdm`) | agent | mostly done | 2026-03-08 | core implemented, close parity |
| `rdiffusion` | Phase 4 | parity comparison script | agent | done | 2026-03-08 | `Study2/verify_rdist_fastdm.py` |
| `rdiffusion` | Phase 5 | combined panel visualization | agent | done | 2026-03-08 | `Study2/figures/rdiffusion_fastdm_compare_panel.png` |
| `rdiffusion` | Phase 6 | plain-language code comments | agent | done | 2026-03-08 | comments added in `gen_data_rdist_fastdm.py` |
| `gen_rand_data` | Phase 1 | source mapping + call graph | agent | done | 2026-03-09 | mapped `hddm/generate.py` -> `kabuki.generate.gen_rand_data` -> `gen_rts` -> `hddm.wfpt.gen_rts_from_cdf` (default) / `_gen_rts_from_simulated_drift` |
| `gen_rand_data` | Phase 2 | python API skeleton | agent | done | 2026-03-09 | created `Study2/gen_data_hddm.py` and `Study2/verify_hddm_gen.py` with preprocessing/validation scaffold |
| `gen_rand_data` | Phase 3 | numerical core implementation | agent | done | 2026-03-20 | `execute=True` now simulates RT/response with subject noise, variability (`sv/sz/st`), methods (`cdf/cdf_py/drift`), and outlier injection |
| `gen_rand_data` | Phase 4 | parity tests vs HDDM outputs | agent | done | 2026-03-20 | verified in `hddm-shell` container (`--size 4000 --subjs 12 --with-hddm-ref`); baseline/variability/scale-invariance deltas logged (largest q90 gap in scale_invariance ~= 0.149) |
| `gen_rand_data` | Phase 5 | visualization panel | agent | done | 2026-03-20 | generated `Study2/figures/gen_rand_data_hddm_compare_panel.png` (hist+density and cumulative quantiles) |
| `gen_rand_data` | Phase 6 | docs + handoff update | agent | done | 2026-03-20 | tracker/session/files updated; open gap recorded below |

Status legend: `not started` | `in progress` | `blocked` | `mostly done` | `done`

## Standard Workflow (For Each Target)

### Phase 0: Define success

1. Pick one target function.
2. Define parity criteria before coding:
   - interface parity (args/defaults),
   - behavior parity (edge cases/outputs),
   - numerical parity (stats + distribution shape).
3. Set acceptable tolerance.

### Phase 1: Source mapping (no coding)

1. Collect wrapper, helpers, low-level kernels, tests, and docs.
2. Build call graph from entry point to lowest-level sampler/math core.
3. Extract parameter semantics: scaling, transforms, constraints, RNG behavior.
4. Record links and key function names in this file.

### Phase 2: API scaffold

1. Build Python skeleton with matching API shape.
2. Implement parameter recycling and preprocessing.
3. Implement validation checks.

### Phase 3: Numerical implementation

1. Port base algorithm first.
2. Add variability/extensions next.
3. Keep function boundaries close to source design.

### Phase 4: Parity testing

1. Compare reference vs Python implementation.
2. Report at least:
   - mean RT,
   - RT variance,
   - response proportion,
   - quantiles (q10, q30, q50, q70, q90).
3. Test baseline, scale-invariance, and variability parameter sets.
4. Add larger-sample stability check.

### Phase 5: Visual validation

1. Create one combined panel:
   - row 1: histogram + density overlays,
   - row 2: cumulative quantile plots (q10, q30, q50, q70, q90).
2. Save in `Study2/figures/` with stable naming.
3. Keep final figure only; remove obsolete outputs.

### Phase 6: Documentation and handoff

1. Update this file with requests, implementation summary, references, results, and open gaps.
2. Update tracker status/date fields.

## Reusable Per-Target Checklist

```text
Target: <function_name>
Owner: <agent/user>
Start date: <YYYY-MM-DD>
Status: <not started|in progress|blocked|mostly done|done>

Phase 0 (goal/tolerance):
- [ ] done

Phase 1 (source mapping):
- [ ] wrapper located
- [ ] low-level kernel located
- [ ] call graph recorded

Phase 2 (API scaffold):
- [ ] skeleton created
- [ ] parameter transforms added
- [ ] validation checks added

Phase 3 (numerical core):
- [ ] base algorithm ported
- [ ] optional variability/extensions ported

Phase 4 (parity tests):
- [ ] summary stats compared
- [ ] quantiles compared
- [ ] edge-case behavior compared

Phase 5 (visualization):
- [ ] combined panel generated

Phase 6 (documentation):
- [ ] URLs and files logged
- [ ] final status updated
```

## Session Log (Condensed)

1. Reverse engineered `rtdists::rdiffusion` and mapped backend stack.
2. Drafted then completed `Study2/gen_data_rdist_fastdm.py` fastdm-style Python implementation.
3. Added parity comparison script: `Study2/verify_rdist_fastdm.py`.
4. Compared Python outputs against R `rtdists` outputs across multiple parameter sets.
5. Added and refined visualization to a single combined panel figure.
6. Added plain-language explanatory comments in the Python implementation.
7. Completed Phase 1 source mapping for `gen_rand_data` (HDDM), including verified call graph, parameter semantics, and external reference mapping.
8. Completed Phase 2 API scaffold for `gen_rand_data` with parameter normalization, validation, and parity-check skeleton scripts.
9. Added root `docker-compose.yml` quick-start services (`hddm-shell`, `hddm-jupyter`) for stable HDDM container workflow.
10. Implemented Phase 3 for `gen_rand_data` in `Study2/gen_data_hddm.py` with full `execute=True` simulation path.
11. Upgraded Phase 4 script (`Study2/verify_hddm_gen.py`) to run multi-scenario parity summaries and optional HDDM reference deltas.
12. Added Phase 5 panel generator (`Study2/phase5_gen_rand_data_hddm_panel.py`) and saved combined panel figure to `Study2/figures/gen_rand_data_hddm_compare_panel.png`.
13. Ran full Phase 4 HDDM-reference parity in Docker (`hddm-shell`) with `--size 4000 --subjs 12 --with-hddm-ref`; confirmed Phase 4 completion and recorded deltas.
14. Updated Phase 5 panel to compare all candidate methods (`cdf`, `cdf_py`, `drift`) against HDDM reference and fixed legend/title overlap for readability.
15. Refactored `Study2/phase5_gen_rand_data_hddm_panel.py` into modular plotting helpers and regenerated a cleaner panel with deduplicated legend items and improved spacing.
16. Realigned `gen_rand_data` panel styling to match `rdiffusion_fastdm_compare_panel.png` conventions (subplot sizing, axis labels, legend style, histogram+density overlays, cumulative-quantile panel format).
17. Synced Phase 5 scenario coverage with Phase 4 (`baseline`, `variability`, `scale_invariance`) and made HDDM reference visually unique (dashed, thicker line style in legends and curves).
18. Added fail-fast guard in `Study2/phase5_gen_rand_data_hddm_panel.py` to prevent silent overwrite by non-HDDM local runs; script now errors unless HDDM reference is present or `--allow-missing-hddm` is explicitly set.
19. Added reusable HDDM synthetic data export pipeline (`Study2/export_hddm_synthetic_data.py`) and cache-first loading in Phase 5 panel script (`Study2/data/hddm_synthetic/*.csv`) so local plotting can reuse pre-generated HDDM references.
20. Streamlined shared scenario/cache logic into `Study2/gen_rand_data_shared.py`, refactored compare/export/plot scripts to reuse it, and cleaned transient `Study2/.DS_Store`, `Study2/data/.DS_Store`, and `Study2/__pycache__/`.
21. Consolidated four HDDM reverse-engineering helper scripts into `Study2/verify_hddm_gen.py` (modes: `parity`, `export`, `panel`) so only two main files remain: `gen_data_hddm.py` and `verify_hddm_gen.py`.
22. Unified script naming for continuity: `gen_data_rdist_fastdm.py`, `verify_rdist_fastdm.py`, `gen_data_hddm.py`, `verify_hddm_gen.py`; older helper filenames in items 12/15/18/19/20 are historical and now superseded by `verify_hddm_gen.py`.

## Canonical Scripts (Current)

- rdist/fastdm generation: `Study2/gen_data_rdist_fastdm.py`
- rdist/fastdm verification: `Study2/verify_rdist_fastdm.py`
- HDDM-generation reverse engineering: `Study2/gen_data_hddm.py`
- HDDM verification/export/panel: `Study2/verify_hddm_gen.py` (`--mode parity|export|panel`)

## Open Gap + Exact Next Step

- Open gap: calibrate simulation core to reduce remaining distribution gap in `scale_invariance` (largest current q90 absolute difference ~= 0.149 vs HDDM reference).
- Exact next step:

```bash
docker compose run --rm hddm-shell python Study2/verify_hddm_gen.py --mode parity --size 8000 --subjs 20 --with-hddm-ref
```

- Then tune method-specific step size/noise handling in `Study2/gen_data_hddm.py` and re-run parity until chosen tolerance is met.

## Next Session Quick Start (Docker)

Run from repo root (`DesignSpace/`):

```bash
# open interactive shell in HDDM container
docker compose run --rm hddm-shell

# inside container: quick sanity checks
python -c "import hddm; print(hddm.__version__)"
python Study2/gen_data_hddm.py
python Study2/verify_hddm_gen.py --mode parity --with-hddm-ref
python Study2/verify_hddm_gen.py --mode export --size 3000 --subjs 8
python Study2/verify_hddm_gen.py --mode panel --size 3000 --subjs 8
```

Reusable cache workflow:

```bash
# 1) export HDDM synthetic references once (Docker)
docker compose run --rm hddm-shell python Study2/verify_hddm_gen.py --mode export --size 3000 --subjs 8

# 2) generate comparison panel locally with cached HDDM references
python Study2/verify_hddm_gen.py --mode panel --size 3000 --subjs 8
```

Jupyter option:

```bash
docker compose --profile jupyter up hddm-jupyter
```

## References

- `https://github.com/rtdists/rtdists/blob/master/R/diffusion.R`
- `https://github.com/rtdists/rtdists`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/R/diffusion.R`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/src/FCalculator.h`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/src/CDF_no_variability.h`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/src/CDF_sz_variability.h`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/src/CDF_sv_variability.h`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/src/CDF_st0_variability.h`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/src/Distribution.h`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/src/Density.h`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/src/Parameters.h`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/src/Sampling.h`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/src/RFastDM.cpp`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/src/RcppExports.cpp`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/R/RcppExports.R`
- `https://raw.githubusercontent.com/rtdists/rtdists/master/man/Diffusion.Rd`
- `https://api.github.com/repos/rtdists/rtdists/git/trees/master?recursive=1`
- `https://github.com/hddm-devs/hddm`
- `https://github.com/hddm-devs/hddm/blob/master/hddm/generate.py`
- `https://raw.githubusercontent.com/hddm-devs/hddm/master/hddm/generate.py`
- `https://raw.githubusercontent.com/hddm-devs/hddm/master/hddm/simulators/basic_simulator.py`
- `https://raw.githubusercontent.com/hddm-devs/hddm/master/hddm/simulators/hddm_dataset_generators.py`
- `https://raw.githubusercontent.com/hddm-devs/hddm/master/src/wfpt.pyx`
- `https://raw.githubusercontent.com/hddm-devs/hddm/master/src/pdf.pxi`
- `https://raw.githubusercontent.com/hddm-devs/hddm/master/src/integrate.pxi`
- `https://raw.githubusercontent.com/hddm-devs/hddm/master/hddm/tests/test_generate.py`
- `https://raw.githubusercontent.com/hddm-devs/hddm/master/hddm/utils.py`
- `https://hddm.readthedocs.io/en/latest/hddm.html`
- `https://hddm.readthedocs.io/en/latest/hddm.simulators.html`
- `https://github.com/hddm-devs/kabuki`
- `https://raw.githubusercontent.com/hddm-devs/kabuki/master/kabuki/generate.py`
- `https://github.com/lnccbrown/ssm-simulators`

## Functions Tracked

- `rdiffusion` (target 1)
- `gen_rand_data` (target 2)

## Files Generated

- `Study2/AGENTS_GEN_MODELS.md`
- `Study2/gen_data_rdist_fastdm.py`
- `Study2/verify_rdist_fastdm.py`
- `Study2/figures/rdiffusion_fastdm_compare_panel.png`
- `Study2/gen_data_hddm.py`
- `Study2/verify_hddm_gen.py`
- `Study2/figures/gen_rand_data_hddm_compare_panel.png`
- `Study2/data/hddm_synthetic/` (reusable HDDM reference CSVs)
- `docker-compose.yml`
