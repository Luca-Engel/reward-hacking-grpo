# Publication checklist

Things that must be true before this repository is made public or cited as a portfolio piece. Nothing here changes the
pre-registered design; it lists facts that could not be verified while the code was written.

## References that must be checked against the source

| reference | where | what is unverified |
|---|---|---|
| Wong et al., "Steering RL Training: Benchmarking Interventions Against Reward Hacking" (LessWrong / AF) | README, DESIGN §1, `CITATION.cff` | full author list, posting date, URL |
| ICLR 2026 workshop version, "Mitigating Reward Hacking with RL Training Interventions" | README, DESIGN §1 | title, authors, whether it is the same work |
| `ariahw/rl-rewardhacking` (code) | README, DESIGN §1 | repository owner and name |
| "rebound" paper, arXiv 2604.01476 | DESIGN §1 | identifier, title, authors, the claim attributed to it |
| Baker et al. 2025; Anthropic 2025; ImpossibleBench (Zhong et al. 2025) | README, DESIGN §1 | exact titles and venues |

Replace every "verify before citing" note once checked (grep for `verify`, `unverified`, `to be verified`).

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
