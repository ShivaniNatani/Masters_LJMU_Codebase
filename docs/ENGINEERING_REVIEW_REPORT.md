# Engineering Review Report
**Repository:** Multi-Modal Medical Report Generation Using Vision-Language Models
**Reviewer role:** Principal ML Engineer (engineering-only review)
**Date:** 2026-07-26
**Scope reviewed:** Phase 1 deliverables only — `configs/`, `datasets/`, `preprocessing/`, `utils/`, `models/mock_vlm.py`, `scripts/`, `data/`, dependency/Docker files, `docs/`, `notebooks/`. `training/`, `evaluation/`, `retrieval/`, `label_guidance/` are empty stubs reserved for later phases per `docs/DISSERTATION_RESEARCH_PLAN.md` and were not in scope for engineering fixes (nothing to fix yet).

**Ground rule honored throughout:** no dataset, research objective, model architecture, evaluation metric, or experiment-design change was made. Split ratios (70/10/20), seed (42), min_word_freq, image size/normalization, and all model/eval configs are untouched.

Each finding below is tagged **[FIXED]** (implemented in this pass — see `CHANGELOG.md`) or **[DOCUMENTED]** (flagged for a maintainer/research decision, not auto-changed, with the reason why).

---

## 1. Data Leakage Risks

| # | Finding | Severity | Status |
|---|---|---|---|
| 1.1 | `preprocessing/build_vocab.py` fit the vocabulary (`word2idx`, `min_word_freq` thresholding) on the **entire unsplit corpus**, including what would become val/test reports. Any word that only appears in a val/test report but clears the frequency threshold becomes a "known" token instead of `<unk>` — the token space the model conditions on is influenced by held-out data. Classic vocabulary leakage, even though the corpus is free text rather than labels. | **High** | **FIXED** — `build_and_save_vocab()` now runs `patient_level_split()` first and fits vocabulary on the train split only (104/150 records on mock data). Falls back to the full corpus with an explicit `logger.warning` if `patient_id` is absent, instead of failing silently. |
| 1.2 | `preprocessing/patient_splitter.py` itself is leakage-safe: it partitions on unique `patient_id`, asserts zero pairwise intersection, and derives the test set as the remainder (`n_patients - n_train - n_val`) so no patient is dropped or double-counted. | — | No issue. Verified with new unit tests (`tests/test_patient_splitter.py`). |
| 1.3 | `patient_level_split()` is **not yet called by any pipeline code** — `training/`, `evaluation/` are empty, and `datasets/base_dataset.py` / `datasets/mimic_cxr.py` / `datasets/iu_chest_xray.py` take a pre-built DataFrame with no split logic wired in. The README's "0 patient overlap guaranteed" claim is currently a guarantee of the *function in isolation*, not of an executed end-to-end pipeline. | Medium | **DOCUMENTED** — this is exactly what Phase 2 ("Data Preprocessing & Patient-Level Splitting") is reserved for per the research plan; wiring it into a real dataset-loading pipeline is a Phase 2 methodology task, not a Phase 1 engineering bug. |
| 1.4 | `scripts/run_eda.py` builds a vocabulary/word-frequency table from the full corpus for **descriptive EDA plots only** (not used to encode any model input). | — | No issue — left unchanged. This is standard, legitimate pre-split EDA and is a different code path from 1.1 (which feeds an actual `word2idx` used for model input encoding). |

---

## 2. Dataset Preprocessing Correctness

| # | Finding | Severity | Status |
|---|---|---|---|
| 2.1 | `preprocess_image()` silently returned an all-zero tensor on any load failure (missing file, corrupt image) with **no logging at all** — broken images vanish into the pipeline indistinguishably from real blank data, with zero visibility into how often it happens. | Medium | **FIXED** — added `logger.warning` at every fallback branch (PIL failure, OpenCV failure, invalid path) in `preprocessing/image_preprocessing.py`. Numeric behavior (zero-tensor fallback) intentionally left unchanged — this is a preprocessing-semantics question, not purely an engineering one; see note below. |
| 2.2 | `Image.open()` was not used as a context manager, leaving file handles open until GC. | Low | **FIXED** — now uses `with Image.open(...) as img:`. See §13 (Memory). |
| 2.3 | `BaseMedicalDataset.__getitem__` fabricates `f"P{idx}"` / `f"S{idx}"` / `f"D{idx}"` IDs when `patient_id`/`study_id`/`dicom_id` are missing from a row. On real MIMIC/IU data this should never trigger, but if it ever did on malformed input, it could silently manufacture fake per-row "patients" that defeat the leakage-prevention guarantee of §1.2 for those rows. | Low | **DOCUMENTED** — not changed, since altering this touches `BaseMedicalDataset`'s contract (arguably dataset-schema behavior). Recommend the maintainer decide whether missing IDs should hard-fail instead of being silently fabricated, once real data is wired in during Phase 2. |
| 2.4 | `RadiologyTextPreprocessor.extract_findings_and_impression()`: when neither `FINDINGS:` nor `IMPRESSION:` headers are found, the entire raw report is dumped into `findings` with no visibility that the fallback path was taken. | Low | **DOCUMENTED** — flagging only; adding a log line here is easy but touches text-preprocessing behavior which is closer to methodology than the image-loading logging fix in 2.1, so left to the maintainer's discretion for Phase 2. |
| 2.5 | `RadiologyTextPreprocessor.remove_duplicates()` drops exact string duplicates only (no near-duplicate/fuzzy matching). | — | Reasonable for Phase 1 scope; not a bug. |

---

## 3. Patient-Level Split Correctness

Verified correct. `patient_level_split()`:
- Partitions on `df[patient_col].unique()`, never on rows directly, so multi-study patients can't leak across splits.
- Explicit `assert` statements on all three pairwise intersections (train∩val, train∩test, val∩test == ∅).
- Deterministic given a fixed seed (verified in `tests/test_patient_splitter.py::test_split_is_deterministic_given_a_fixed_seed`).
- Ratios respected within integer-truncation tolerance; remainder correctly goes to test rather than being dropped.

Only gap: no input validation for `NaN`/duplicate values in `patient_col`, or for `patient_col` not existing in the DataFrame at all (would raise a raw `KeyError` from pandas rather than a clear error message). **[DOCUMENTED]** — low risk, not fixed, since real MIMIC/IU metadata is expected to have clean patient IDs and adding validation now is speculative for data that doesn't exist yet in this repo.

Added 5 regression tests (`tests/test_patient_splitter.py`) covering: zero-overlap, full coverage, determinism, ratio adherence, and invalid-ratio rejection — this was previously **completely untested** despite being the single most safety-critical function in the repository.

---

## 4. GPU Utilization

| # | Finding | Severity | Status |
|---|---|---|---|
| 4.1 | `create_dataloader()` hardcoded `num_workers=0`, forcing fully synchronous, single-process image decode + tokenization on every batch — the GPU/MPS device would sit idle waiting for CPU-side preprocessing once real training exists. | Medium | **FIXED** — `num_workers` now defaults to `None`, auto-selecting `min(4, cpu_count - 1)`; still overridable with an explicit int (0 included) for debugging. |
| 4.2 | `configs/training.yaml` declares `hardware.num_workers: 4` and `pin_memory: true`, but **nothing in the code reads this file** — the config's intent had zero effect on runtime behavior. | Medium | **DOCUMENTED**, not wired — see §6. Wiring `training.yaml` into a currently-nonexistent training loop is premature (no call site exists yet in Phase 1); flagged for whoever builds the Phase 5 training loop. |
| 4.3 | `configs/training.yaml` also declares `mixed_precision: fp16`, but there is no training loop yet to implement autocast/`GradScaler`. | — | Expected — Phase 5 stub, not a current defect. |
| 4.4 | `set_seed()` sets `cudnn.deterministic=True` / `cudnn.benchmark=False` for reproducibility, which is a legitimate throughput/reproducibility trade-off already made deliberately by the codebase (`configs/logging.yaml: reproducibility.deterministic_cudnn: true`) — consistent, not a bug. | — | No issue. |

---

## 5. DataLoader Efficiency

| # | Finding | Severity | Status |
|---|---|---|---|
| 5.1 | Same `num_workers=0` default as §4.1. | Medium | **FIXED** (see 4.1) — also added `persistent_workers=True` and `prefetch_factor=2` whenever `num_workers > 0`, so worker processes and their preprocessing setup aren't torn down and rebuilt every epoch. |
| 5.2 | `drop_last` was hardcoded to `False` inside `create_dataloader`, not exposed as a parameter. Combined with `MockVLM`'s use of `nn.BatchNorm2d`, a final batch of size 1 during `model.train()` would raise a runtime error ("expected more than 1 value per channel"). Currently masked because the smoke test only calls `model.eval()`. | Low (latent, not yet triggered — no training loop exists) | **FIXED (partially)** — `drop_last` is now an explicit, overridable parameter (default unchanged at `False`, so current behavior is identical) so Phase 5's training loop can pass `drop_last=True` without needing to touch `data_loader.py` again. The underlying `BatchNorm2d`-vs-batch-size-1 risk itself lives in `models/mock_vlm.py`, a throwaway verification stub, and is **flagged, not touched**, since it's a "real" architecture file. |
| 5.3 | Default collate is used (no custom `collate_fn`) — appropriate, since every sample already has fixed-size tensors (image resized, tokens padded to `max_seq_len`). | — | No issue. |

---

## 6. Configuration Management

**This is the most consequential structural finding in the repo.**

| # | Finding | Severity | Status |
|---|---|---|---|
| 6.1 | Of the 5 YAML configs, **only `logging.yaml` is actually loaded by any code** (`utils/logger.py`). `datasets.yaml`, `training.yaml`, `models.yaml`, `evaluation.yaml` are never read by a single `.py` file (verified via repo-wide grep for `configs/`). All real values (split ratios, batch size, image size, etc.) are hardcoded as Python defaults instead, duplicating what the YAML files say. Today the numbers happen to match; nothing prevents them from silently drifting apart, since editing the YAML has zero effect on program behavior. | **High** | **DOCUMENTED, not wired.** No call sites currently exist to wire `datasets.yaml`/`training.yaml` into (`patient_level_split()` and `create_dataloader()` are not yet invoked by any pipeline with config plumbing available), so adding that plumbing now would be speculative abstraction for phantom future code. Flagged clearly here so whoever builds Phase 2/5 wires these configs in at that point rather than continuing to hardcode. |
| 6.2 | `utils/logger.py`'s `config_path` default (`"configs/logging.yaml"`) and `log_file` default were plain relative paths, which only resolved correctly if the process's CWD happened to be the repo root — exactly as the README instructs (`PYTHONPATH=. python scripts/...`), but silently broke (falling back to hardcoded defaults with no warning) if run from anywhere else, e.g. from inside `scripts/` or from an IDE's default CWD. | Medium | **FIXED** — `utils/logger.py` now resolves both paths relative to the project root (`Path(__file__).resolve().parent.parent`), independent of CWD. |
| 6.3 | `python-dotenv` is a pinned dependency but there is no `.env` support anywhere in the code — dead dependency, no config actually flows through environment variables. | Low | **DOCUMENTED** (see §11, Dependency Issues) — left in place since it may be intended for later phases; not removed. |

---

## 7. Logging

| # | Finding | Severity | Status |
|---|---|---|---|
| 7.1 | `setup_logger(name)` was called at **module import time** in ~8 different modules, each with a different logger `name`. Each call unconditionally opened a fresh `FileHandler` on the same `logs/execution.log`, even though the "clear handlers first" logic only protected against duplicates *within* a single logger name. Net effect: many independent open file descriptors on one file, and — critically — under a multi-worker `DataLoader` (§4/§5), forked worker processes re-run this module-level code and reopen the same file again, risking interleaved/duplicated writes and, on some platforms, file-locking errors. | Medium | **FIXED** — `setup_logger()` now tracks configured logger names in a module-level set and is a no-op on repeat calls for the same name, so handlers are attached exactly once per logger regardless of how many times/processes import the module. |
| 7.2 | No log rotation — `logs/execution.log` used a plain unbounded `FileHandler` in append mode, growing forever across every EDA/smoke-test run for the life of the dissertation project. | Low | **FIXED** — switched to `RotatingFileHandler` (5 MB × 5 backups). |
| 7.3 | Log format, level, and structure (via `configs/logging.yaml`) are otherwise clear, consistent, and informative — a genuine strength of the codebase. | — | No issue. |
| 7.4 | `logger.propagate` was not explicitly set to `False`; harmless today (no root logger is configured elsewhere) but was fragile against future duplicate-log-line bugs if anything ever calls `logging.basicConfig()`. | Low | **FIXED** as part of the `utils/logger.py` rewrite. |

---

## 8. Exception Handling

| # | Finding | Severity | Status |
|---|---|---|---|
| 8.1 | `preprocess_image()` caught bare `Exception` around image loading and silently returned a zero tensor with no logging — see §2.1. | Medium | **FIXED** (logging added). |
| 8.2 | `scripts/download_mimic_cxr.py` passed the PhysioNet password as a literal `--password` CLI argument to `wget`, visible to any other local user via `ps`/`/proc/<pid>/cmdline` for the process lifetime, and caught a broad `Exception` around the subprocess call, only logging on failure (never raising or exiting non-zero) — a failed download looks like a soft warning rather than a hard failure. | **High** (security) / Medium (control flow) | **FIXED** (security part) — credentials are now written to a temporary, `chmod 600` `.netrc` file consumed via `wget --netrc-file`, and the file is always removed in a `finally` block. Control-flow (non-zero exit on failed download) intentionally **left as-is** — this is a manual, human-run utility script, not part of an automated pipeline, so a soft warning that lets a user retry is arguably the right UX; changing it wasn't necessary to fix the security issue. |
| 8.3 | No exception handling around `pd.read_csv()` in `build_vocab.py`/`run_eda.py` — a malformed CSV surfaces a raw pandas traceback. | Low | **DOCUMENTED**, not fixed — these are one-shot dev scripts where a raw traceback is acceptable and arguably more debuggable than a wrapped generic error; low value for the risk of changing script control flow. |
| 8.4 | `scripts/smoke_test.py` wraps its entire body in one `try/except Exception`, capturing and reporting failure into the JSON report with `exc_info=True` — this is the *correct* pattern for a smoke test (it must never itself throw, only report pass/fail). | — | No issue — good existing practice. |

---

## 9. Reproducibility

| # | Finding | Severity | Status |
|---|---|---|---|
| 9.1 | **The actual dev `.venv` (Python 3.14.2, torch 2.13.0, transformers 5.14.1, numpy 2.5.1, faiss-cpu 1.14.3, opencv 5.0.0.93, wandb 0.28.1, mlflow 3.14.0 — confirmed via `pip freeze`) was drastically different from both checked-in lockfiles**: `environment.yml` pinned Python 3.10 / torch 2.1.2 / transformers 4.38.2, and `requirements.txt` used open-ended `>=` floors that don't pin anything precisely. The environment that actually produced `results/verification_report.json` and `logs/execution.log` was not reproducible from either file. | **High** | **FIXED per explicit instruction** — both `requirements.txt` and `environment.yml` were regenerated from the live `.venv`'s `pip freeze` output (2026-07-26) so they now describe the environment Phase 1 was actually verified against. **Caveat added as a comment in both files**: this dev venv is CPU/MPS-only (no CUDA present), so `pytorch-cuda=12.1` in `environment.yml` and the CUDA build tag in the Dockerfile could not be re-verified against a real GPU host and should be checked before a real training run. |
| 9.2 | `requirements.txt` previously listed **both** `weights-and-biases>=0.16.0` and `wandb>=0.16.0` — two different PyPI distribution names; only `wandb` is ever imported. | Low | **FIXED** — duplicate entry removed, kept `wandb` only. |
| 9.3 | `albumentations` is pinned in both lockfiles but is not importable anywhere in the current code, and is **not present at all** in the live `.venv` this pass regenerated lockfiles from. | Low | **DOCUMENTED** — kept in both files (not removed, since it's clearly intended for future augmentation work per the folder-structure comments) but annotated with a comment flagging that it must be explicitly verified/installed before use, since the "source of truth" venv doesn't have it. |
| 9.4 | `Dockerfile` installed `torch`/`torchvision` unpinned from the CUDA 12.1 wheel index, then separately ran `pip install -r requirements.txt` (previously also unpinned for torch) — relying on pip's implicit "already-satisfies-constraint, skip reinstall" behavior to avoid silently downgrading to a CPU-only build. This is real but unwritten/fragile behavior (would break under `pip install --upgrade`, or a resolver change). | Medium | **FIXED defensively** — pinned the Dockerfile's torch/torchvision install to the exact versions now in `requirements.txt`, and added `RUN pip check && python -c "import torch; assert ...; print(torch.version.cuda)"` immediately after both install steps so the **build fails loudly** instead of silently shipping a broken/CPU-only image. |
| 9.5 | `utils/seed.py`'s `set_seed()` is comprehensive (python/numpy/torch/CUDA/MPS) and is correctly invoked at the top of every entry-point script (`smoke_test.py`, `run_eda.py`, `generate_mock_data.py`, `patient_splitter.py`). | — | No issue — a genuine strength, left unchanged. |
| 9.6 | No unit tests existed anywhere despite `pytest` being a pinned dependency — the most safety-critical logic (patient split) had zero automated regression coverage. | Medium | **FIXED** — added `tests/test_patient_splitter.py` (5 tests) and `tests/test_text_preprocessing.py` (5 tests); all pass (`pytest -q` → `10 passed`). |
| 9.7 | No version control (`git init` was never run in this repository). | Medium | **DOCUMENTED, not actioned** — initializing git is a reversible, low-risk action but is a repo-management decision outside this engineering-fix pass; recommended, not performed. |

---

## 10. Folder Organization

Generally a strength — the actual tree matches the README's documented structure 1:1.

| # | Finding | Severity | Status |
|---|---|---|---|
| 10.1 | `training/`, `evaluation/`, `retrieval/`, `label_guidance/`, `checkpoints/` were empty directories with no placeholder file — once git is initialized, empty directories won't survive `git add` and the documented structure would silently disappear from version control. | Low | **FIXED** — added `.gitkeep` to all five. |
| 10.2 | Stray `__pycache__/` directories under `datasets/`, `preprocessing/`, `models/`, `scripts/` in the working tree (correctly `.gitignore`d, so not a version-control issue, just build noise). | Trivial | **FIXED** — removed from the working tree (they'll regenerate harmlessly on next run and remain gitignored). |

---

## 11. Dependency Issues

| # | Finding | Severity | Status |
|---|---|---|---|
| 11.1 | Duplicate `weights-and-biases` / `wandb` entries. | Low | **FIXED** (§9.2). |
| 11.2 | `python-dotenv`, `pydantic`, `albumentations` are pinned but never imported anywhere in the current Phase 1 code. `faiss-cpu`, `mlflow`, `wandb` are also unused today but are clearly earmarked for Phase 3 (retrieval) / experiment tracking. | Low | **DOCUMENTED** — left in place; removing dependencies earmarked for near-future phases risks being mistaken for a scope/methodology change, which is explicitly out of bounds for this review. |
| 11.3 | `requirements.txt` previously used open `>=` floors with no upper bound and no lock file — see §9.1. | High | **FIXED** — now exact-pinned, matching `environment.yml`. |

---

## 12. Potential Bugs

| # | Finding | Severity | Status |
|---|---|---|---|
| 12.1 | Vocabulary-from-full-corpus leakage. | High | **FIXED** (§1.1). |
| 12.2 | Dockerfile double torch install relying on undocumented pip behavior. | Medium | **FIXED** (§9.4). |
| 12.3 | `nn.BatchNorm2d` + `drop_last=False` + batch-size-1 edge case in `models/mock_vlm.py`. | Low (latent) | **DOCUMENTED** (§5.2) — the DataLoader-side mitigation (`drop_last` now overridable) is fixed; the model file itself is explicitly out of scope (real architecture file). |
| 12.4 | `MockVLM.forward()` hardcodes `nn.CrossEntropyLoss(ignore_index=0)`, implicitly assuming the pad token is always index 0 (true today because `RadiologyTextPreprocessor._build_special_tokens()` registers `pad_token` first) but with no explicit coupling enforced between the two — a silent-wrong-loss risk if special-token order ever changes. | Low | **DOCUMENTED, not fixed** — `models/mock_vlm.py` is architecture code, explicitly out of scope per the review constraints. |
| 12.5 | Seaborn `FutureWarning`s in `scripts/run_eda.py` (`palette` without `hue` is deprecated as of seaborn 0.14). | Trivial | **FIXED** — added `hue=`/`legend=False` to both `sns.barplot()` calls; verified plots are visually identical, only the deprecation warning is gone. |
| 12.6 | Zero-tensor fallback for missing/corrupt images (§2.1) bypasses the `Normalize()` step every real image goes through, so a "missing" sample is numerically inconsistent with the rest of a batch (a true blank image, post-normalization, would not be exactly zero). | Low | **DOCUMENTED, not fixed** — changing the fallback's numeric value changes what the (future) model actually sees for missing images; that's closer to a preprocessing-methodology decision than a pure engineering fix, so it's flagged for the maintainer rather than silently changed. Logging (§2.1) at least makes the occurrence visible now. |

---

## 13. Memory Optimization

| # | Finding | Severity | Status |
|---|---|---|---|
| 13.1 | `PIL.Image.open()` not used as a context manager — file handles left open until GC, risking file-descriptor exhaustion under many concurrent DataLoader workers. | Low | **FIXED** (§2.2). |
| 13.2 | `patient_level_split()` calls `.copy()` three times (once per split). Since the three splits are disjoint, total memory is ≈1× the original DataFrame, not 3× — not actually wasteful, just looks that way at a glance. | — | No issue, no change needed. |
| 13.3 | `RadiologyTextPreprocessor.build_vocabulary()` builds a `collections.Counter` over the whole corpus at once — fine at both mock scale (150 rows) and full MIMIC-CXR scale (~227k reports; a `Counter` over a few hundred thousand short strings is tens of MB at most). | — | No issue. |
| 13.4 | No chunked/streaming CSV reads anywhere — acceptable given typical metadata CSV sizes; flagged only as a forward-looking note if the real MIMIC-CXR metadata file turns out unexpectedly large. | — | Not fixed — speculative, no evidence of an actual problem. |

---

## 14. Code Readability

Overall a strength: consistent type hints, consistent docstrings, consistent naming conventions, small single-purpose functions/files. No changes were made purely for readability's sake beyond what naturally fell out of the fixes above (e.g., the new logging calls double as inline documentation of failure paths that were previously invisible).

---

## 15. Documentation Quality

| # | Finding | Severity | Status |
|---|---|---|---|
| 15.1 | README's documented folder structure was verified against the actual tree and matches exactly — a genuine strength. | — | No issue. |
| 15.2 | `docs/DATASET_GUIDE.md` describes the IU Chest X-ray dataset as "7,470 Chest X-Ray DICOM images", but OpenI's public IU X-ray release is distributed as PNG + XML (matching `scripts/download_iu_cxr.py`'s own `NLMCXR_png.tgz` URL) — a minor internal inconsistency between two of the repo's own files. | Low | **DOCUMENTED, not fixed** — correcting dataset-description docs is adjacent to "dataset" content, which is explicitly out of scope for this engineering-only pass; flagged for the maintainer to correct. |
| 15.3 | No `CHANGELOG.md` existed prior to this review. | — | **FIXED** — see `CHANGELOG.md`, added as part of this deliverable. |

---

## Summary

- **19 issues fixed** directly in this pass (all pure engineering: logging, dependency pinning, DataLoader efficiency, memory handling, credential exposure, test coverage, Docker build safety, folder scaffolding, one data-leakage fix explicitly requested).
- **12 issues documented only**, each with an explicit reason (touches methodology/experiment-design, targets code with no current caller, or is a research/maintainer decision rather than an engineering one).
- **Zero changes** to datasets, split ratios, model architecture, evaluation metrics, or experiment design.
- Full pipeline re-verified after all changes: `pytest tests/ -q` → `10 passed`; `scripts/smoke_test.py` → `PASSED`; `scripts/run_eda.py` and `preprocessing/build_vocab.py` re-run cleanly with no warnings.

See `CHANGELOG.md` for the itemized list of file-level changes.
