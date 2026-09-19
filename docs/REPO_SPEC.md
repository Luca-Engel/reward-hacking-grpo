# REPO_SPEC — layout, schemas, and interface contracts

This is the contract every implementation session must follow so that independently
written modules fit together. `DESIGN.md`/`PREREG.md` define *what*; this defines *where
and in what shape*. If a session must deviate, it documents the deviation in
`docs/SPEC_DEVIATIONS.md` (one line each) instead of silently changing an interface.

## 1. Layout

```
reward-hacking-grpo/
  DESIGN.md PREREG.md BUDGET.md SCHEDULE.md README.md DEVIATIONS.md LICENSE CITATION.cff
  pyproject.toml  uv.lock  .python-version  requirements-gpu.txt
  configs/
    base.yaml
    prompts.yaml                    # hint wordings + candidates; frozen at Gate 1f
    arms/{clean,hackable}_{none,subtle,explicit}.yaml  hackable_subtle_ast.yaml   # 7 arms
  src/rhg/
    config.py  seeds.py  manifest.py  budget.py  runlog.py  plan.py  prereg_constants.py
    e2e_mock.py  check_docs.py
    data/     load.py  tests_split.py  dedupe.py  prompts.py  build.py  fixture.py  explore.py
    env/      extract.py  sandbox.py  grader.py  cache.py  labels.py  monitor.py
    detect/   ast_detector.py
    judge/    rubric.py  client.py  run.py  cost.py
    validate/ metrics.py  harness.py  controls.py  sample.py  label.py  calibrate.py
    train/    run.py  mock_policy.py  trl_config.py  trl_trainer.py  rollout_io.py  watchdog.py  gpu_memory.py
    eval/     generate.py  pass_rate.py  probe_hints.py  bench.py
    analysis/ stats.py  power.py  endpoints.py  robustness.py  quality.py  prereg_check.py
              figures.py  report.py  examples.py  bundle.py  simulate.py  run.py
  scripts/    setup_box.sh smoke.sh bench_throughput.sh measure_pass_rate.sh probe_hints.sh
              pilot.sh run_arm.sh run_all.sh calibrate_judge.sh judge_all.sh analyze.sh
              package_results.sh freeze_prereg.sh  _common.sh
  notebooks/  01_data_exploration.ipynb  (built by notebooks/build_01.py)
  tests/      fixtures/  test_*.py
  docs/       REPO_SPEC.md  dataset_notes.md  detector_notes.md  judge_notes.md  GPU_COMPAT.md
              trl_grpoconfig_fields.json  labeling_protocol.md  SPEC_DEVIATIONS.md
              WRITEUP_TEMPLATE.md  SAFETY_ETHICS.md  MORNING_CHECKLIST.md
  prereg/     FREEZE.json  AMENDMENTS.jsonl  budget_decision.md  hint_selection.json
              pilot_gate.json                             # written at the gates
  data/       raw/ processed/ labels/                  # gitignored except tests/fixtures
  results/    bench/ runs/ judge/ analysis/ notebooks/ probe/ cache/ ledger.jsonl   # gitignored
  results_public/   # shareable bundle built by rhg.analysis.bundle (committed at the end)
```

Python package `rhg` under `src/` (src layout). Python ≥3.12, managed by `uv`.

## 2. Dependencies
- `pyproject.toml` core deps: `pyyaml pydantic numpy pandas scipy matplotlib datasets
  huggingface_hub anthropic tqdm`; dev group: `pytest nbformat nbconvert ipykernel jupyter`.
  `uv sync` on Windows must work with core+dev only.
- **The GPU stack is NOT in `uv.lock`.** `requirements-gpu.txt` (exact pins:
  torch/transformers/trl/peft/vllm/accelerate/etc.) is installed on the box by
  `scripts/setup_box.sh` via `uv pip install -r requirements-gpu.txt`. Heavy imports are
  lazy (inside functions) everywhere.
- Every module that touches the GPU stack has a `--mock`/`mock` path runnable on CPU.

## 3. Config schema (`src/rhg/config.py`, pydantic v2; YAML with `extends`)

`configs/arms/<arm>.yaml` = `extends: base.yaml` + an `arm:` block. **One YAML per arm;
seed is an override**, never in the arm file.

```yaml
# configs/arms/hackable_subtle.yaml
extends: base.yaml
arm: {id: hackable_subtle, reward: hackable, hint: subtle, monitor: null}
# hackable_subtle_ast.yaml: monitor: ast_narrow_penalty
```

```yaml
# configs/base.yaml (all keys required in the resolved config)
run:      {seed: 0, output_root: results/runs, mode: train, confirmatory: false, tag: null,
           step_timeout_s: 900}
model:    {name: Qwen/Qwen3-1.7B, enable_thinking: false, dtype: bfloat16}
lora:     {r: 32, alpha: 32, dropout: 0.0, target: all-linear}
grpo:     {max_steps: 100, prompts_per_step: 16, gens_per_prompt: 8, lr: 7.0e-5,
           beta: 0.0, loss_type: dr_grpo, max_prompt_tokens: 768,
           max_completion_tokens: 1024, grad_checkpointing: true, vllm_gpu_mem_util: 0.35}
sampling: {temperature: 1.0, top_p: 1.0, top_k: -1}
data:     {processed_dir: data/processed, k_reward_tests: 5, max_heldout_tests: 20,
           band_low: 0.10, band_high: 0.40}
reward:   {monitor_penalty: 1.5}
eval:     {val_every: 20, val_samples_per_problem: 4, test_samples_per_problem: 8,
           xhint_samples_per_problem: 4}
sandbox:  {timeout_s: 6.0, mem_mb: 2048, workers: 0, cache: true}   # workers 0 = auto (cores-2)
budget:   {usd_per_hour: 0.45, ledger: results/ledger.jsonl, stop_at_usd: 28.0}
judge:    {model: claude-haiku-4-5-20251001, votes: 2, third_vote_on_disagree: true,
           max_usd: 4.0, max_flagged_per_run: 100, audit_frac: 0.05}
```
`grpo.loss_type` etc. are mapped to the pinned TRL `GRPOConfig` by
`train/trl_trainer.py`; a CPU test must assert every mapped key exists in the installed
`trl.GRPOConfig` signature (skip only if `trl` is not installed).

Overrides: `--seed N`, `--set a.b.c=value` (repeatable; YAML-parsed values).
`load_config(arm, overrides) -> Config`. Two hashes (sha256 of canonical JSON):
- `config_hash`: resolved config **without** `run.seed`, `run.output_root`, `run.mode`,
  `run.tag` (identical across seeds of an arm);
- `run_hash`: includes `run.seed`.
`run_id = f"{arm.id}__s{seed}"`. Run dir = `results/runs/<run_id>/`.

## 4. Run manifest (`results/runs/<run_id>/manifest.json`, written at start, updated at end)
```json
{"run_id": "", "arm": "", "seed": 0, "config_hash": "", "run_hash": "",
 "git_sha": "", "git_dirty": false, "git_diff_sha256": null,
 "prereg_tag_present": false, "prereg_tag_is_ancestor": false,
 "freeze_json_sha256": null, "split_hash": "", "prompts_hash": "", "dataset_revision": null,
 "libs": {"python": "", "torch": null, "transformers": null, "trl": null, "peft": null,
          "vllm": null, "datasets": null, "numpy": null, "anthropic": null},
 "hardware": {"gpu_name": null, "gpu_count": 0, "driver": null, "cuda": null, "cpu_cores": 0,
              "hostname_sha256": ""},
 "mode": "train|mock", "confirmatory": false,
 "started_at": "", "finished_at": null, "wall_s": null, "usd_per_hour": 0.0, "usd": null,
 "status": "running|completed|invalid|failed", "invalid_reason": null}
```
`--confirmatory` requires clean git tree, tag `prereg-v1` an ancestor of HEAD, and
`FREEZE.json` hashes matching; else refuse to start.

## 5. Data schemas

`data/processed/problems.jsonl` (one per admitted problem):
`problem_id, source, difficulty, tags, date, description, starter_code, import_prefix,
entry_point, reference_solution, reward_tests[], heldout_tests[], cluster_id, p_A, p_B_full,
p_B_visible, split (train|val|test)`. A test is `{"id": int, "src": "assert candidate(...) == ...",
"kind": "assert"}` (executable in a namespace where `candidate` is the entry point).

`data/processed/splits.json`: lists per split + `split_hash` (sha256 of sorted ids+split).
`data/processed/passrate_A.jsonl` / `passrate_B.jsonl`: per problem counts.

## 6. Rollout / step / eval logs (per run dir)

`rollouts.jsonl.gz`, one JSON per rollout:
```json
{"run_id":"","phase":"train|eval_val|eval_test|eval_test_xhint","step":0,"eval_hint":null,
 "problem_id":"","sample_idx":0,
 "completion":"","n_tokens":0,"truncated":false,"code_extracted":true,
 "reward":0.0,
 "labels":{"defines_rt":false,"rt_ok":false,"visible_pass":false,"heldout_pass":false,
           "timeout":false,"crash":false,"hack_rt":false,"attempt_rt":false,
           "correct":false,"gap_other":false},
 "monitor":{"ast_narrow":false,"ast_broad":false,"ast_categories":[]}}
```
`steps.jsonl`: `step, reward_mean, loss, grad_norm, completion_len_mean, truncation_rate,
frac_zero_adv_groups, hack_rt_rate_train, attempt_rt_rate_train, correct_rate_train,
t_gen, t_reward, t_train, t_sync, t_step, tokens_gen, tokens_train`.
`evals.json`: per eval point, per-problem counts of every label. `status.json`,
`config.resolved.yaml`, `manifest.json`, `stdout.log`.

Budget ledger `results/ledger.jsonl`: `{ts, kind: train|bench|pilot|passrate|probe|judge|other,
run_id, wall_s, usd_per_hour, usd, note}`.

Judge output `results/judge/<run_id>.jsonl`: `run_id, problem_id, sample_idx, step, votes[],
label, inclusion_prob, source: ast_flagged|audit, tokens_in, tokens_out, usd`.

Analysis outputs `results/analysis/`: `per_seed.csv`, `tests.json` (every test with p,
min_attainable_p, family, CONFIRMATORY|EXPLORATORY), `tables/*.csv`, `figures/*.png`,
`REPORT.md`. Every figure title/table caption carries the CONFIRMATORY/EXPLORATORY stamp.

## 7. CLI contracts (`python -m ...`; all accept `--help`; all have `--mock` where GPU/API is used)
- `rhg.train.run --arm A --seed N [--mock] [--set k=v] [--confirmatory] [--steps N]`
- `rhg.data.build --stage {fetch,validate,tests,split} [--fixture]`
- `rhg.eval.pass_rate --stage {A,B} [--mock]`  (GPU; writes passrate_*.jsonl)
- `rhg.eval.probe_hints [--mock]`  (GPU; base-model ATTEMPT_RT per wording; writes `results/probe/`)
- `rhg.eval.bench [--mock]`  (GPU; writes `results/bench/throughput.json`)
- `rhg.budget {check --next-run-usd X | record ... | cost_model}`
- `rhg.judge.run --runs ... [--dry-run|--estimate-only]`
- `rhg.validate.{harness,label,sample}`
- `rhg.analysis.{run,prereg_check,power,simulate}`
- `rhg.plan {list,shard,ladder,replacement,next}`;  `rhg.check_docs`
- `rhg.e2e_mock`  (whole pipeline on CPU with the mock policy; must finish in < 10 min)
- `rhg.validate.calibrate`, `rhg.analysis.{robustness,bundle}` (also via `run`), `rhg.data.dedupe`
- Exit codes: 0 ok, 2 usage/config error, 3 guard refused (budget/prereg), 75 infrastructure
  stall (watchdog; replaceable), 1 other.

## 8. Testing rules
- `uv run pytest -q` must be green on Windows with core+dev deps only, no network, no GPU,
  no API keys. Network- or GPU-dependent tests are marked and skipped by default.
- Sandbox tests execute only benign code plus hand-written *toy exploit* snippets that
  are safe (no file deletion, no network, no process spawning).
- Every module in §1 has at least one unit test. Statistics code is tested against
  brute-force enumeration.
