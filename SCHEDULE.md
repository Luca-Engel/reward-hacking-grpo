# SCHEDULE — day by day, with go/no-go gates

Your time: ~2–3 days. GPU spend is metered by `results/ledger.jsonl`. "NO-GO" means stop
spending and decide (options are pre-declared where they exist).

## Day 0 — overnight (no GPU, no money)
Everything on Day 0 runs on CPU with mock policies.

**Gate 0 (morning, ~45 min of your time)** — GO iff all hold:
- `uv run pytest -q` green; `uv run python -m rhg.e2e_mock` completes (22 mock runs →
  analysis → report) and recovers the planted effect.
- `notebooks/01_data_exploration.ipynb` executed headless on the fixture (see
  `results/notebooks/`).
- You have read `DESIGN.md`/`PREREG.md`, the detector/judge code (drafts were not
  reviewed), and `docs/GPU_COMPAT.md` (risk notes on the TRL/vLLM stack).
Do not rent a GPU until green.

## Day 1 — GPU box, ≈$5–6, ~6 h of your time
1. **Setup + smoke** (`scripts/setup_box.sh`, `scripts/smoke.sh`): 5 GRPO steps end to end.
   - **Gate 1a** — GO iff no hang, finite reward, ≤15 min, phase timings logged. Otherwise
     follow `docs/GPU_COMPAT.md` fallbacks; **cap debugging at $1**, then NO-GO (rethink the
     trainer; do not spend the main budget).
2. **Throughput bench** (`scripts/bench_throughput.sh`) → `rhg.budget cost_model`.
   - **Gate 1b** — GO iff `main_usd ≤ 16` and `wall_h ≤ 12` at 22 runs, else apply the ladder
     (BUDGET §4) and write `prereg/budget_decision.md`. NO-GO if even the floor (11 runs)
     does not fit.
3. **Base pass rate + splits** (`scripts/measure_pass_rate.sh`, then `rhg.data.build --stage split`):
   - **Gate 1c** — GO iff ≥150 train / ≥40 val / ≥60 test problems, ≥95% of references
     valid, and the labeler agrees with hand-inspected synthetic controls. Else the single
     pre-declared band widening `[0.05,0.50]`; else NO-GO.
4. **Hint calibration probe** (`scripts/probe_hints.sh`; base model, no training, ≈$0.3):
   - **Gate 1d** — GO iff a subtle candidate meets the rule (DESIGN §2.3) and step-0
     ATTEMPT_RT is ordered none < subtle < explicit with non-overlapping Wilson CIs. Else
     NO-GO, with exactly one further round (≤$0.3) before freezing, in the direction the
     probe reports: the pre-declared weaker wordings W1–W3 (`--round weaker`) if S1 fails from
     above, else ≤3 new stronger wordings S4–S6 (`--round stronger`).
5. **Pilots** (`scripts/pilot.sh`; seeds 9000+): `hackable_explicit` 60 steps; `clean_subtle` 30 steps.
   - **Gate 1e** — GO iff `hackable_explicit` shows train HACK_RT ≥ 0.10 by step 60, no infra
     fault, and `clean_subtle` HACK_RT ≈ 0 with the labeler behaving. If no emergence: **one**
     allowed retry with lr×2 (record it); second failure is NO-GO. Inspect ~20 pilot
     rollouts by eye for definitional artifacts (honest self-tests, timeouts).
6. **Judge calibration** (`scripts/calibrate_judge.sh --yes`, on your **local** machine, ≈$0.1):
   the real judge on the ≥30 synthetic controls.
   - **Gate 1e2** — GO iff ≥90% overall agreement, ≥90% recall on override-type controls and
     ≤10% false positives on honest controls. Otherwise edit the rubric (controls only,
     never real rollouts) and rerun; the rubric hash is frozen at the next step.
7. **Freeze** (`scripts/freeze_prereg.sh`): requires the artifacts of steps 2–6, writes
   `prereg/FREEZE.json`, commits, tags `prereg-v1`.
   - **Gate 1f** — GO iff `python -m rhg.analysis.prereg_check` passes **and you have pushed
     the commit and tag to your public remote** (`git push origin HEAD; git push origin
     prereg-v1`). The public timestamp is what makes the pre-registration checkable; set up
     the boxes from a checkout that contains the tag.

## Day 2 — main runs, ≈$16 GPU, mostly waiting (~3 h of your time)
`scripts/run_all.sh [--shard i/n]` on 1–3 boxes: priority order, resume, budget guard,
health dashboard. Nobody looks at hack-rate-by-arm.
- **Gate 2a** (after first 2 primary pairs): health only; if mean `run_usd` > 1.25× projection,
  apply the ladder now.
- **Gate 2b** (60% of runs or budget): recompute projection; apply ladder if needed.
- `scripts/package_results.sh` pulls run directories back to your local machine. Shut boxes
  down immediately after (idle time is unmetered).

## Day 3 — local, ≈$4.5 API, ~6 h of your time
1. **Judge** (`scripts/judge_all.sh`, runs locally, key never on the GPU box).
   - **Gate 3a** — pre-flight cost from measured token counts ≤ $4 or subsample with weights.
2. **Human labels** (`python -m rhg.validate.label`, ~40 items + 10 repeats; ~45 min, labels
   entered before viewing any detector/judge output).
3. **Validation report** (execution-anchored precision/recall with CIs; human stratum
   secondary).
   - **Gate 3b** — informational. A poor judge changes what secondary claims are allowed,
     not the primary endpoint.
4. **Analysis run** (`scripts/analyze.sh`): `prereg_check` must pass for CONFIRMATORY labels;
   otherwise output is stamped EXPLORATORY. Then figures, `results/REPORT.md`, `python -m rhg.analysis.bundle --out results_public/`,
   and the write-up from `docs/WRITEUP_TEMPLATE.md`.
   Include the min-attainable-p table, `DEVIATIONS.md`, the invalid-run list, and the
   ladder state actually executed.
