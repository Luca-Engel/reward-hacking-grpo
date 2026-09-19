# SPEC_DEVIATIONS

One line per deviation from `docs/REPO_SPEC.md` or the design docs made during implementation.

- 01: `extends: base.yaml` in `configs/arms/*.yaml` (as REPO_SPEC §3 writes it) is resolved relative to the extending file first, then falls back to the config root, since base.yaml lives in `configs/`, not `configs/arms/`.
- 01: `load_config` also requires `arm.id` in the (pre-override) arm file to equal the arm's file name, and falls back to the repo-root `configs/` when a relative `config_dir` does not exist under the cwd.
- 01: Config numeric fields are strict-typed (no bool/str coercion; an int for a float field is normalised to float) and `grpo.loss_type` is restricted to {grpo, bnpo, dr_grpo, dapo}; subtask 10 must revisit the set if the pinned TRL differs.
- 01: The override value `1e-4` is parsed as a float (PyYAML alone reads it as a string); `run.seed` from `--seed` wins over a `run.seed=` override.
- 01: Config and Grpo blocks expose `rollouts_per_step`, and `sampling` exposes `top_k_enabled`, as read-only properties (not part of the dump/hash).
- 02: In the manifest, `git_sha` and `git_dirty` are `null` (unknown) when git is unavailable, and `split_hash`/`prompts_hash` are `null` when `splits.json`/`prompts.yaml` are absent, instead of `""`/`false` (an unknown state must never read as "clean").
- 02: `git_dirty` counts non-ignored untracked files; `git_diff_sha256` is the sha256 of `git diff HEAD` (tracked changes only). `prompts_hash`, code-group and requirements hashes normalise CRLF to LF so Windows and Linux checkouts agree.
- 02: `prereg_check.check` adds an item `freeze_matches_tag` (working `FREEZE.json` must equal the version committed at `prereg-v1`) so the freeze cannot be silently rewritten after tagging; `write_freeze` refuses once the tag exists. Amendments must chain from the frozen hash (`old_hash` = currently expected hash) or the group fails as "chain broken".
- 02: Amendment `group` values are the bare code-group names (`analysis`, `env`, `detect`, `judge_rubric`, `data_build`) or `config:<arm>`, `prompts`, `hint_selection`, `hyperparameters`, `split`, `dataset_revision`, `requirements_gpu`; `--amend` amends every drifted key unless `--group` is given.
- 02: Extras beyond the subtask text: `rhg.budget status`, `prereg_check --write-freeze/--gpu-type/--force`, `cost_model --t-step-source {formula,measured,max}` (default `formula` = BUDGET §2 exactly; `measured` uses the median of `step_times_s[2:]`), optional `cache_hit_rate`/`exec_s_per_rollout_uncached` in `throughput.json`. `cost_model` refusing to overwrite the decision file exits 3; a computed "below the 11-run floor" outcome exits 0 and is stated in the output and decision draft.
- 02: The ledger guard's "recommended ladder step" = highest-priority `LADDER` step whose run count fits (distinct `train` run ids in the ledger) + floor((stop_at - spent) / next_run_usd); it is advisory and only phrases the refusal.
