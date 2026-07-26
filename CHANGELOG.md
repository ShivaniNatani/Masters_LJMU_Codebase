# Changelog

All notable engineering changes from the 2026-07-26 principal-ML-engineer review are listed here.
Full rationale and severity ratings for each item live in `docs/ENGINEERING_REVIEW_REPORT.md`.

No datasets, research objectives, model architecture, evaluation metrics, or experiment design
(split ratios, seed, hyperparameters) were changed. All changes are engineering-only.

## [Unreleased] - 2026-07-26

### Fixed

- **Data leakage — vocabulary building** (`preprocessing/build_vocab.py`): vocabulary is now
  fit on the patient-level TRAIN split only (via `patient_level_split()`), instead of the full
  pre-split corpus. Falls back to the full corpus with an explicit warning if `patient_id` is
  absent from the input CSV.
- **Reproducibility — environment drift**: `requirements.txt` and `environment.yml` regenerated
  from the actual dev `.venv` (`pip freeze`, Python 3.14.2) to match the environment the Phase 1
  smoke test and EDA pipeline were actually verified against. Both previously described a
  materially different, stale environment (Python 3.10, torch 2.1.2, transformers 4.38.2, `>=`
  floors with no lock). Removed a duplicate `weights-and-biases`/`wandb` dependency entry.
  CUDA/GPU-specific pins (`pytorch-cuda=12.1`) could not be re-verified against a real GPU host
  from this CPU/MPS-only dev environment — flagged with an inline comment.
- **Dockerfile — silent CUDA-build downgrade risk**: pinned the explicit CUDA 12.1 torch/torchvision
  install to the exact versions in `requirements.txt`, and added a `pip check` + Python import
  assertion immediately after both install steps so the image build fails loudly instead of
  potentially shipping a CPU-only PyTorch build silently.
- **Logging — duplicate file handlers & unbounded log growth** (`utils/logger.py`): `setup_logger()`
  is now idempotent per logger name (previously reopened a `FileHandler` on `logs/execution.log`
  every call/import, including inside forked DataLoader worker processes). Switched from a plain
  `FileHandler` to a `RotatingFileHandler` (5 MB × 5 backups). Logger paths (`configs/logging.yaml`,
  `logs/execution.log`) now resolve relative to the project root instead of the process's CWD, so
  logging no longer silently breaks when a script isn't run from the repo root. Set
  `logger.propagate = False` to prevent future duplicate-log-line bugs.
- **Silent failures in image loading** (`preprocessing/image_preprocessing.py`): `preprocess_image()`
  now logs a warning on every fallback path (PIL failure → OpenCV fallback → zero-tensor fallback),
  previously completely silent. Missing-image numeric fallback behavior (zero tensor) is unchanged.
- **Resource handling** (`preprocessing/image_preprocessing.py`): `Image.open()` now used as a
  context manager, releasing the file handle immediately instead of leaving it open until GC.
- **DataLoader efficiency / GPU utilization** (`datasets/data_loader.py`): `num_workers` now
  auto-selects `min(4, cpu_count - 1)` instead of a hardcoded `0` (fully synchronous loading);
  added `persistent_workers=True` and `prefetch_factor=2` whenever `num_workers > 0`. `drop_last`
  is now an explicit, overridable parameter (default unchanged: `False`) so a future training loop
  can avoid a `BatchNorm2d`-vs-batch-size-1 crash without further changes to this file.
- **Credential exposure** (`scripts/download_mimic_cxr.py`): PhysioNet password is no longer passed
  as a `wget --password` CLI argument (visible via `ps`/`/proc` to any local user for the process
  lifetime). Credentials are now written to a temporary, owner-only-readable (`chmod 600`) `.netrc`
  file consumed via `wget --netrc-file`, always deleted in a `finally` block.
- **Deprecation warnings** (`scripts/run_eda.py`): added `hue=`/`legend=False` to both `sns.barplot()`
  calls to resolve seaborn `FutureWarning`s about `palette` without `hue`; plot output is visually
  unchanged.
- **Folder scaffolding**: added `.gitkeep` to `training/`, `evaluation/`, `retrieval/`,
  `label_guidance/`, `checkpoints/` so these documented-but-empty directories survive a future
  `git init`/`git add`. Removed stray `__pycache__/` build artifacts from the working tree
  (already `.gitignore`d; harmless regen).

### Added

- `tests/test_patient_splitter.py` — 5 regression tests for `patient_level_split()` covering
  zero patient overlap, full row/patient coverage, determinism under a fixed seed, ratio
  adherence, and invalid-ratio rejection. This was previously the single most safety-critical
  function in the repo with zero automated test coverage.
- `tests/test_text_preprocessing.py` — 5 tests for `RadiologyTextPreprocessor` covering
  encode/decode round-trips, unknown-token → `<unk>` handling, `min_word_freq` filtering,
  findings/impression section extraction, and exact-duplicate removal.
- `docs/ENGINEERING_REVIEW_REPORT.md` — full review report (this pass), covering all 15
  requested review dimensions with severity ratings and fixed-vs-documented status per finding.
- `CHANGELOG.md` — this file.

### Documented only (not changed — see report for rationale)

- `patient_level_split()` is not yet wired into any dataset-loading pipeline (Phase 2 scope).
- `datasets.yaml` / `training.yaml` / `models.yaml` / `evaluation.yaml` are parsed by no code
  today (only `logging.yaml` is read) — flagged as a configuration-drift risk for whoever wires
  Phase 2/5/6 code against these files, rather than adding speculative plumbing now.
- `BaseMedicalDataset.__getitem__`'s silent ID fabrication for missing `patient_id`/`study_id`/
  `dicom_id`.
- `models/mock_vlm.py`'s `BatchNorm2d`-vs-batch-size-1 latent risk and hardcoded
  `ignore_index=0` coupling — explicitly out of scope as architecture code.
- Zero-tensor (non-normalized) fallback for missing images — a preprocessing-semantics decision,
  not purely an engineering one.
- `docs/DATASET_GUIDE.md`'s "DICOM" vs. actual PNG format description for the IU Chest X-ray
  dataset.
- No version control (`git init` was never run) — recommended, not actioned.

### Verification

- `pytest tests/ -q` → `10 passed`
- `PYTHONPATH=. python scripts/smoke_test.py` → `PHASE 1 SMOKE TEST PASSED SUCCESSFULLY`
  (`results/verification_report.json` status: `PASSED`)
- `PYTHONPATH=. python scripts/run_eda.py` → completes with no warnings
- `PYTHONPATH=. python preprocessing/build_vocab.py` → builds vocab from train-split-only
  (104/150 mock records), confirmed via log output
- `python -m py_compile` on every changed `.py` file → no syntax errors
