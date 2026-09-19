# SPEC_DEVIATIONS

One line per deviation from `docs/REPO_SPEC.md` or the design docs made during implementation.

- 01: `extends: base.yaml` in `configs/arms/*.yaml` (as REPO_SPEC §3 writes it) is resolved relative to the extending file first, then falls back to the config root, since base.yaml lives in `configs/`, not `configs/arms/`.
- 01: `load_config` also requires `arm.id` in the (pre-override) arm file to equal the arm's file name, and falls back to the repo-root `configs/` when a relative `config_dir` does not exist under the cwd.
- 01: Config numeric fields are strict-typed (no bool/str coercion; an int for a float field is normalised to float) and `grpo.loss_type` is restricted to {grpo, bnpo, dr_grpo, dapo}; subtask 10 must revisit the set if the pinned TRL differs.
- 01: The override value `1e-4` is parsed as a float (PyYAML alone reads it as a string); `run.seed` from `--seed` wins over a `run.seed=` override.
- 01: Config and Grpo blocks expose `rollouts_per_step`, and `sampling` exposes `top_k_enabled`, as read-only properties (not part of the dump/hash).
