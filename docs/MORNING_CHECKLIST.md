# Morning checklist (SCHEDULE Day 0, Gate 0)

The overnight run built everything on CPU with mock policies. **Nothing GPU-side has ever run**, no paid API was called, and
several assumptions below were made blind. Do not rent a GPU until Part A is green and you have read Part B.

## Part A: Gate 0, exact commands (~45 min of your time)

Run from the repository root (PowerShell or Git Bash; `uv` on PATH).

```bash
# 1. repository state
git status --short && git log --oneline | head -20

# 2. clean install and static checks
uv sync                                          # core + dev only; the GPU stack is NOT installed here
uv run python -m compileall src -q
uv run python -m rhg.check_docs                  # docs and code agree (exit 0; accepted deviations are listed)

# 3. the whole suite (~15 min on a 16-core laptop: see "Suite runtime" below)
uv run pytest -q

# 4. the whole pipeline on CPU with the mock policy: 22 runs -> analysis -> report -> bundle
uv run python -m rhg.e2e_mock --quick            # ~1.5 min: must end with "E2E OK"
uv run python -m rhg.e2e_mock                    # ~3 min, full-size runs (SCHEDULE Gate 0); tree in results/e2e/
#   inspect: results/e2e/analysis/REPORT.md, results/e2e/analysis/examples.md, results/e2e/results_public/

# 5. the data notebook, executed headless on the fixture
uv run python notebooks/build_01.py --execute --fixture --fixture-gates   # -> results/notebooks/01_executed.ipynb

# 6. dry-run the GPU scripts (no GPU, no spend): they print what they would do
scripts/setup_box.sh --dry-run
scripts/smoke.sh --dry-run
scripts/bench_throughput.sh --usd-per-hour 0.45 --dry-run
scripts/run_all.sh --dry-run
```

Then read, in this order: `DESIGN.md`, `PREREG.md`, `docs/GPU_COMPAT.md` (TRL/vLLM risk notes, ranked fallbacks), the AST detector
(`src/rhg/detect/ast_detector.py`, `docs/detector_notes.md`) and the judge (`src/rhg/judge/`, `docs/judge_notes.md`): the drafts of
detector and judge were **not** reviewed before the overnight run (DESIGN §12.7). `docs/SPEC_DEVIATIONS.md` is the list of every
place the build deviated from the spec, grouped by area.

**Gate 0 is GO iff:** `pytest` and `check_docs` are green, both `e2e_mock` runs end with `E2E OK`, the notebook
executed, and you have read the items above. Anything else: fix before spending.

### Decide before Gate 1f (freeze); none of it is an outcome-dependent choice

- `configs/base.yaml`: `budget.usd_per_hour` = the rate you actually pay (manifests and the config hash use it);
  `grpo.lr` only if the single allowed lr x 2 pilot retry passed (record it).
- `configs/prompts.yaml`: set `subtle_selected` to the wording `prereg/hint_selection.json` names.
- Review the judgement calls the build made where the docs are silent (all logged in `docs/SPEC_DEVIATIONS.md`): TRL
  `scale_rewards="group"` with `loss_type=dr_grpo` (trl-config), constant lr schedule, `beta=0`; the Gate 1e criterion
  "clean_subtle mean training HACK_RT <= 0.02" (plan); the replacement-seed rule (plan); the H2 caveat that HACK_RT is mechanically
  coupled to problem difficulty (stats). Changing them after the freeze needs an amendment.

### Suite runtime

The whole suite is slower than the 5-minute target of the ground rules (~13.5 min once the analysis package existed, 14 min 25 s
for 1296 tests on a clean copy; details in `docs/SPEC_DEVIATIONS.md`, e2e entries). `uv run pytest -q tests/test_e2e_quick.py tests/test_check_docs.py` is the fast
integration check; `uv run pytest -q -x` stops at the first failure. Network/GPU/API/slow tests are skipped by default
(`-m network`, `-m gpu`, `-m api`, `-m slow` opt in).

## Part B: assumptions made blind, and the first symptom to look for

`UNVERIFIED` = never run against the real thing. "First symptom" is what you will see first if the assumption is wrong.

| # | Assumption (where) | First symptom when it breaks | First thing to do |
|---|---|---|---|
| 1 | **GPU stack pins** install and initialise together: torch 2.13.0 (CUDA 13 build), vllm 0.28.0, transformers 5.15.0, trl 1.13.0, peft 0.20.0 (`requirements-gpu.txt`; only solver-checked, `docs/GPU_COMPAT.md` §1-2) | `uv pip install -r requirements-gpu.txt` resolver error, or `import vllm` / `LLM(...)` fails with a CUDA-driver or `libcudart` message (driver < 580), or `Engine core initialization failed` | GPU_COMPAT §6 fallbacks in order: lower `grpo.vllm_gpu_mem_util`, sleep mode, the cu129 / older pinned set. Edit `requirements-gpu.txt` only before the freeze |
| 2 | **TRL argument mapping** (`train/trl_config.py`, `trl_trainer.py`; tested only against recorded field lists and stubs, `tests/stubs/`) | `TypeError: unexpected keyword` from `GRPOConfig`, or the smoke run fails at step 1 with "step without a logged loss/grad_norm" (`logging_steps=1` is required), or no `steps.jsonl` line and the watchdog exits 75 | Read `results/smoke/*/stdout.log` and `status.json.phase`; `--set run.step_timeout_s=300` for the smoke run; re-dump fields with `python src/rhg/train/dump_trl_fields.py` |
| 3 | **Estimand-relevant TRL defaults** pinned by judgement: `scale_rewards="group"`, `vllm_importance_sampling_correction=False`, constant lr, `beta=0`, `mask_truncated_completions=False` (SPEC_DEVIATIONS 10) | Reward curves that do not move in the smoke/pilot, or `grad_norm` 0 / exploding, `frac_zero_adv_groups` near 1 | Compare with the Dr. GRPO variant on the pilot only, and decide before the freeze |
| 4 | **TRL internals used for timing** (`trainer.vllm_generation.generate/.sync_weights` wrapped on the instance; `on_log` fires after `on_step_end`) (11) | `trl_timing.jsonl` shows `gen_measured=false` / `sync_measured=false`, `t_gen`/`t_sync` = 0, or a step is logged twice | The timings fall back to differences; check the bench's `timers_measured` before trusting `BUDGET_MEASURED.md` |
| 5 | **vLLM adapter evals**: a fresh `VLLMGenerator(adapter_path=...)` per snapshot with a `LoRARequest`; `LLM`/`SamplingParams` kwarg names written from memory (`eval/generate.py`, stubs marked UNVERIFIED) (06, 11) | `TypeError` on engine construction; **or eval hack rates that are identical at every snapshot** (the adapter was not applied: eval = base model); `gpu_memory_not_released` after a snapshot eval; 6 engine loads per run make `t_eval` large | In the smoke run compare `eval_val` at step 0 vs step 5 on a `hackable_explicit` pilot; check `evals.json` per-snapshot differ; `EVAL_GPU_MEM_UTIL` in `trl_trainer.py` |
| 6 | **bf16 merge/unmerge drift** while TRL syncs LoRA to vLLM (GPU_COMPAT §4, trl#6688 open) | Sampling policy differs from the trained one: eval hack rate systematically below the training-rollout hack rate at the same step | Compare `hack_rt_rate_train` (last steps) with the val eval at that step in the pilot |
| 7 | **LeetCodeDataset conversions**: `check(candidate)` parsed into per-assert tests, imports, `Solution().method` entry points, dedupe/clustering, reference validation in the sandbox (04, `docs/dataset_notes.md`; only the synthetic fixture was run) | `rhg.data.build --stage tests` drops many problems (`data/processed/tests_report.json`), `validate` reports `reference_validity` < 95% or many `reference_timeout` drops (machine-speed dependent: build once, copy `problems.jsonl`), fewer than 150 / 40 / 60 train / val / test problems at Gate 1c | Read `tests_report.json`, `validation_report.json`, `docs/dataset_notes.md`; then the single pre-declared `--widen` |
| 8 | **Base pass-rate measurement** assumes the fallback prompt renderer matches Qwen3's chat template and the same code extraction as the grader (06) | Stage A pass rates near 0 or 1 for most problems, `no-code` share high, or `PromptTooLongError` at the start of a run (a prompt within ~50 tokens of `max_prompt_tokens` plus a hint) | Print a rendered prompt next to the real tokenizer's `apply_chat_template(..., enable_thinking=False)`; check `passrate_A_stats.jsonl` |
| 9 | **Linux sandbox limits** (`RLIMIT_AS` 2 GB etc.) were only exercised on Windows (03) | Every rollout `crash=True` or `timeout=True` in the first smoke `evals.json`; `MemoryError` on import of common libraries; labels all false | `--set sandbox.mem_mb=4096` for a diagnosis (then it is a freeze decision); inspect `stdout.log` of a graded completion by hand |
| 10 | **Reward execution is CPU-bound**: needs >= 8 physical cores (BUDGET §2) | The bench shows `t_reward` dominating `t_step` | Rent a box with more cores or accept the ladder; `sandbox.workers` |
| 11 | **Cost model**: every $/step and $/run figure is a planning prior until `results/bench/throughput.json` exists (BUDGET) | `BUDGET_MEASURED.md` says `main_usd` > 16 or `wall_h` > 12 | Apply the pre-declared ladder (`rhg.budget cost_model`), write `prereg/budget_decision.md` |
| 12 | **Judge prices and Batch API shapes** (Haiku 4.5 $1/$5 per MTok, batch 50%, cache read 0.1x, 4096-token cache minimum; request/response shapes) come from the `claude-api` skill's tables and a fake `anthropic` module (07, `docs/judge_notes.md`) | `scripts/calibrate_judge.sh --yes` raises a request-shape error; the real spend differs from the pre-flight estimate by more than ~2x (compare `ledger.jsonl` kind `judge` with the Anthropic console); the rubric never hits the prompt cache | Run calibration first (~$0.1-0.2); recompute `cost.py` inputs from the measured token counts before `scripts/judge_all.sh --yes` |
| 13 | **Judge recall on the exploit**: the rubric avoids the literal `run_tests`, so recall is unmeasured until calibration (07, 08) | Gate 1e2 fails (< 90% override-type recall or > 10% honest false positives) | Edit the rubric using the synthetic controls only, never real rollouts; the rubric hash freezes at Gate 1f |
| 14 | **Mock dynamics are my modelling choices, not measurements** (`train/mock_policy.py`, SPEC_DEVIATIONS 09, 16): emergence, displacement and dose-response in the e2e run say nothing about Qwen3-1.7B | A real pilot with no emergence while the e2e run "recovers" the plant | That is the pre-mortem's first risk: Gate 1d / 1e exist for it; do not adjust the analysis |
| 15 | **Gate 1d/1e thresholds** implemented from prose (12): pilot emergence = trailing-5-step training HACK_RT >= 0.10 by step 60; `clean_subtle` <= 0.02 | A pilot that visibly hacks by eye but fails the gate (or the reverse) | Inspect ~20 pilot rollouts by hand (SCHEDULE Gate 1e); the rule is not to be changed after seeing outcomes |
| 16 | **Watchdog and step timeout** (`run.step_timeout_s` 900; exit 75 = infrastructure) (09, 12) | First step legitimately slow (vLLM compile / weight load) and killed as a stall | Longer `--set run.step_timeout_s=` for the smoke run only |
| 17 | **Grading cache** keyed on code + tests + grader version (03): safe by construction, but a grader edit without a `GRADER_VERSION` bump would serve stale results | Labels that do not change after fixing the harness | Bump `GRADER_VERSION` in `rhg/env/grader.py`; delete `results/cache/grade/` |
| 18 | **Contamination proxy** horizon `2024-06-30` for Qwen3 is assumed, not published (15) | (only labelled UNVERIFIED in the notebook) | Read it as a proxy |
| 19 | **Tokenizer-free fixtures**: notebook and mock use a word-count token proxy unless a tokenizer is in the local HF cache (15) | Token statistics look low | Pre-download the tokenizer (not weights) on the local machine |

If a GPU-side item (1-6, 8-10) fails, follow `docs/GPU_COMPAT.md` §6 and cap debugging at **$1** (SCHEDULE Gate 1a); then NO-GO and rethink
the trainer rather than spending the main budget.

## Part C: things that are deliberately left for the real runs

`prereg/` gate artifacts, `BUDGET_MEASURED.md`, `results_public/`, the human labels (`python -m rhg.validate.label`), the README
provenance table (`{{PREREG_COMMIT}}` and friends) and `docs/WRITEUP_TEMPLATE.md` placeholders are filled in only when the corresponding
real step has happened.
