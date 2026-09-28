# Safety and ethics

This project trains a small open-weights model in a setting where it is *paid* for exploiting its own grader,
and then studies how often and how visibly it does so. That is the whole scope; the notes below say what that
does and does not involve.

## Model-written code is untrusted code

- Every completion is executed, so every completion is treated as hostile. Execution happens in a fresh
  subprocess (no network, cleared environment, temporary working directory, wall-clock timeout, and on Linux
  rlimits) started by `rhg.env.sandbox`. Results come back over a nonce'd channel so that model code cannot
  forge a verdict. The Windows development path enforces only the wall-clock timeout and says so at import.
- Training and evaluation with real model output run **only on a rented, disposable GPU box** (see
  `scripts/setup_box.sh`) that holds no personal files and **no secrets**: the judge API key never exists on
  that machine (`scripts/judge_all.sh` runs locally, after the run directories have been copied back), and the
  sandbox environment is cleared of everything except what execution needs.
- The box is destroyed after the runs. Nothing model-written is executed on a personal machine except the
  hand-written toy exploit snippets in the test suite (no file deletion, no network, no process spawning,
  no writes outside temp directories).
- The exploit under study (`run_tests()` overwrite) is a benign, well-known test-harness exploit. Sandbox
  escapes are not the subject, and any residual exploit surface is tracked only as `VISIBLE_ONLY` (DESIGN §8.11).

## No real-world uplift

- The model is Qwen3-1.7B with LoRA adapters; the environment is competitive-programming problems; the
  exploit gives no capability beyond "make a unit-test harness report success". Nothing here produces or
  improves a technique against a deployed system.
- Findings are about *this recipe on this split* (DESIGN §9). They are not evidence about frontier models and
  must not be quoted as such.

## What is and is not released

Released (the `results_public/` bundle, built by `python -m rhg.analysis.bundle`, which refuses to publish if
its secret scan finds anything):

- the report, all tests with exact p-values and minimum attainable p, per-seed tables, figures;
- per-run manifests and resolved configs (hostnames only as hashes), the pre-registration freeze and the
  amendment log, the spend summary by kind, the validation report;
- a seeded random sample of final-evaluation rollouts per arm and a random, non-cherry-picked examples gallery.

Not released:

- trained adapters and full raw rollouts (they are large and add nothing that the sample does not show;
  available on request from the archived run directories if that ever becomes appropriate);
- API keys, absolute paths, hostnames, `.env` files, cache directories, the human-labelling scratch files;
- any prompt or completion the author judges harmful on inspection (none is expected; the gallery is
  reviewed before publishing).

## Honest reporting

The pre-registration tag is pushed to a public remote before any main run, deviations are logged with a
timestamp and printed in the report, and null or uninterpretable outcomes ("no discovery", "inconclusive at
this power") are reported under those exact names. Human labelling is a single-rater sanity layer and is
described as such.

The project spends the author's own money only (hard ceiling $30, enforced by a ledger and launch guards).
