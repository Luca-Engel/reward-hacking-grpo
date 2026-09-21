# Publication checklist

Things that must be true before this repository is made public or cited as a portfolio piece. Nothing here changes the
pre-registered design; it lists facts that could not be verified while the code was written.

## References — verification status (checked 2026-09-21 against the live sources)

| reference | where | status |
|---|---|---|
| Wong, Engels & Nanda, "Steering RL Training: Benchmarking Interventions Against Reward Hacking" (LessWrong / AF, 2025-12-29) | README, DESIGN §1, `CITATION.cff` | **Verified.** Full author list, date and URL confirmed against the live LessWrong/AF post. |
| ICLR 2026 workshop version, "Mitigating Reward Hacking with RL Training Interventions" | README, DESIGN §1, `CITATION.cff` | **Verified.** Same three authors, confirmed via the ICLR 2026 virtual program page (workshop: "Principled Design for Trustworthy AI"). Same project as the LessWrong post; title differs slightly. |
| `ariahw/rl-rewardhacking` (code) | README, DESIGN §1 | **Verified.** Owner `ariahw`, repo `rl-rewardhacking`, confirmed on GitHub. |
| Baker et al. 2025, "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation" | README, DESIGN §1, `CITATION.cff` | **Verified.** arXiv:2503.11926, Bowen Baker + 8 co-authors, 2025-03-14. |
| Anthropic 2025, "Natural Emergent Misalignment from Reward Hacking in Production RL" | README, DESIGN §1, `CITATION.cff` | **Verified.** arXiv:2511.18397, 21 authors incl. Monte MacDiarmid, Evan Hubinger, Sam Bowman. |
| ImpossibleBench, Zhong, Raghunathan & Carlini 2025 | README, DESIGN §1, `CITATION.cff` | **Verified.** arXiv:2510.20270, "ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases," 2025-10-23, ICLR 2026. |
| "rebound" paper, Wu & Tang, arXiv 2604.01476 | DESIGN §1, `CITATION.cff` | **Mostly verified.** Identifier and authors (Rui Wu, Ruixiang Tang) confirmed; the claim attributed to it (hack attempts fail, retreat, then succeed) matches the actual paper. **Still open:** the paper's own title is inconsistent between sources — its arXiv abstract page (both v1 and v2, fetched 2026-09-21) gives "When Reward Hacking Rebounds: Understanding and Mitigating It with Representation-Level Signals," while the rendered HTML page's own `<title>` gives "From Rebound to Remedy: Understanding and Mitigating Reward Hacking via Representation Engineering." Re-check https://arxiv.org/abs/2604.01476 directly at cite time and use whichever title it shows then. |

All notes above except the rebound paper's title have been resolved; grep for `verify`, `unverified`, `to be verified` should now only surface that one item.

## Placeholders

- `CITATION.cff`: `authors[0].name` (currently a username), and the commented `repository-code`, `date-released`, `commit` fields.
- `LICENSE`: confirm the copyright holder.
- README "Results": replace the "no real run has happened" statement only with numbers produced by `rhg.analysis.run`.

## Pre-registration provenance

1. `scripts/freeze_prereg.sh`, then push the commit and the `prereg-v1` tag (`scripts/run_all.sh` refuses to launch unless `origin`
   holds the same tag object).
2. Deposit the tagged commit with an independent registry (an OSF registration, or a Zenodo release whose DOI timestamps the
   archive) and record the URL in `DEVIATIONS.md`. Git dates are author-controlled; the registry timestamp is what a third party can check.
3. Delete local backup branches and check `git log --format='%ae %ce'` shows only the address you want public before the first push.

## Things not yet validated on real hardware

- `rhg.train.trl_trainer` was written against the published TRL/vLLM sources and has only run against stub packages. `scripts/smoke.sh`
  (Gate 1a) is the first real test; until it passes, no statement about the trainer's behaviour is a measurement.
- The Linux sandbox limits (`RLIMIT_AS/CPU/FSIZE`) have not been exercised on this project's data; grade pass rates on Linux
  (Gate 1c FAILs on files graded without enforced limits).
