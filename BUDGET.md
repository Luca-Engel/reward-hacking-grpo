# BUDGET — cost model and cut rules

Hard ceiling **$30** (GPU rental + API). Compute is the binding constraint on the design.
All *planning-prior* numbers below are **unmeasured guesses used only to choose defaults**;
Gate 1b replaces them with numbers from `results/bench/throughput.json` via
`python -m rhg.budget cost_model` (writes `BUDGET_MEASURED.md`). Nothing here is
"derived from measurement" until that file exists.

## 1. Envelope

| line | cap (USD) |
|---|---|
| GPU: setup + smoke + throughput bench | 1.5 |
| GPU: base pass-rate measurement (stages A+B) | 1.0 |
| GPU: hint probe + pilots | 2.5 |
| **GPU: main runs (22 planned)** | **16.0** |
| API: judge (cap enforced in code) | 4.0 |
| API: dev/validation/synthetic-control calls (incl. judge calibration ≈0.1) | 0.5 |
| contingency (infra replacement runs only) | 2.5 |
| unmetered slack (idle box time, storage, egress) | 2.0 |
| **total** | **30.0** |

Launch guard: `run_all.sh` refuses to start any run if `metered_spend + projected_run_cost
> 28.0` (ceiling minus unmetered slack). The ledger (`results/ledger.jsonl`) is appended
after every metered action (train/bench/pilot/passrate/judge) with wall-seconds ×
`$/hour`, or API token cost.

## 2. Cost model (formulas; inputs come from measurement)

```
t_step   = t_gen + t_reward + t_train + t_sync          # median over steps 3..N of the bench
t_gen    = gen_tokens_per_step / gen_tok_per_s          # vLLM colocated, measured
t_reward = rollouts_per_step * exec_s_per_rollout / n_cpu_workers
t_train  = train_tokens_per_step / train_tok_per_s      # LoRA fwd+bwd, grad ckpt, measured
t_sync   = LoRA merge/sync to vLLM per step, measured
run_s    = T * t_step + n_eval * t_eval + t_startup      # n_eval = 5 val evals + 1 test eval
run_usd  = run_s / 3600 * usd_per_hour * (1 + 0.08)      # 8% idle/overhead margin
main_usd = sum over arms of seeds * run_usd              # all arms priced equal
wall_h   = (22 * run_s / 3600) / n_boxes                 # boxes run shards in parallel
```

**Planning prior (unmeasured):** 4090 24 GB at $0.35–0.55/h; `t_step` 35–70 s at 128
rollouts × ≤1024 tokens ⇒ 65–125 min/run ⇒ **$0.45–1.00/run ⇒ $10–22 for 22 runs**.
This is why the cut ladder exists. An A100 costs more per hour but is ~2–3× faster; the
bench decides. Two to three boxes in parallel keep Day 2 wall-clock ≈ 10–12 h at unchanged
cost. `t_reward` is measured by the bench **with the content-addressed grading cache on** (production setting; it caches raw execution results only, never timeouts) and the hit rate is printed; the bench also reports the uncached figure, and an 8-step bench under-estimates the steady-state hit rate, so the projection is conservative. Reward execution needs ≥ 8 physical CPU cores; if the box has fewer, `t_reward`
dominates and is the first thing the bench should reveal.

Judge cost prior: Claude Haiku-class via the Batch API (≈50% discount) with prompt caching
of the rubric; ≈$0.0005–0.001 per call; ≤ ~3–4k items × ≈1.5 votes. Re-derived from
measured token counts on the synthetic-control suite before any judge run (Gate 3a);
the code refuses to exceed the $4 cap and instead subsamples flagged items with recorded
inclusion weights. Verify current prices with the `claude-api` skill when implementing.

## 3. Decision at Gate 1b
1. Compute `main_usd` at full design from measured numbers.
2. If `main_usd ≤ 16.0` **and** `wall_h ≤ 12` (with the boxes you will actually rent) → run
   the full 22.
3. Else apply the cut ladder (§4) top-to-bottom until it fits; record the result in
   `prereg/budget_decision.md` **before** freeze.

## 4. Cut ladder (pre-declared; applied only at Gate 1b/2a/2b, never based on outcomes)

Principle: **fewer non-primary arms at full seeds first; then explicit, stated power loss.
The primary contrast keeps full seeds as long as possible. T is never a cost lever.**

| step | change | runs after | stated consequence |
|---|---|---|---|
| 0 | full design | 22 | — |
| 1 | `hackable_none` 2→1 | 21 | floor check only; H1 none-level has n=1 |
| 2 | drop `clean_none` | 19 | no unhinted clean baseline; `clean_subtle` is the learning-curve reference |
| 3 | drop `hackable_subtle_ast` | 16 | **H4 unevaluated (future work)** |
| 4 | drop `clean_explicit` | 14 | H1 loses its prompt-vs-reward control at the explicit level; H1 becomes "hackable-arm trend, prompt confound unresolved" |
| 5 | `hackable_explicit` 3→2 | 13 | H1 trend has n=2 at the top level |
| 6 | primary 5 v 5 → 4 v 4 | 11 | primary significant only if 4/4 seeds emerge (p=0.0143; 3/4 gives 0.071) |
| floor | — | — | if 11 runs still exceed the cap: **no confirmatory study**; report pilots/probes only |

Launch order encodes the ladder: runs are launched in priority order (tier 0: primary,
interleaved H/C by seed; then `hackable_explicit`, `clean_explicit`, `hackable_subtle_ast`,
`clean_none`, `hackable_none`), so running out of money mid-way cuts the tail automatically
and equals a ladder state.

## 5. Mid-run checks
- **Gate 2a** (after the first 2 primary pairs): if measured mean `run_usd` > 1.25× the
  projection, apply the ladder immediately to the remaining budget.
- **Gate 2b** (at 60% of runs or 60% of the main cap): recompute; apply ladder if needed.
- Contingency $2.5 may only fund replacement runs for invalid runs, up to 3.

## 6. Cheap things that are always allowed to shrink
Eval sizes for val trajectories (never the final test eval), judge subsample (with
weights), number of synthetic controls, notebook size. Not allowed: T, K reward tests,
the final test-eval size, hint wordings after freeze.
