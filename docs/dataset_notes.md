# Dataset notes

Everything below was produced by `python -m rhg.data.build --stage {fetch,tests,validate}` on the real
`newfacade/LeetCodeDataset` on the Windows dev box (CPU only, 16 cores, 14 sandbox workers). Numbers are
copied from `data/processed/tests_report.json` and `validation_report.json` (both gitignored, regenerate with
the commands at the end). The `split` stage needs the GPU-box pass-rate files and has **not** been run on real
data; it is exercised on the synthetic fixture and on synthetic pass-rate files in `tests/test_data.py`.

## 1. Provenance

`data/processed/DATASET_REVISION` (one line, read verbatim by `rhg.manifest`):

```
newfacade/LeetCodeDataset@215604aeed660029df7de2fea5a4d7b6ed476a08 datasets=5.0.1 downloaded=2026-09-19
```

Hub `last_modified` of that commit: 2025-05-29. Files: `LeetCodeDataset-train.jsonl` (2641 rows),
`LeetCodeDataset-test.jsonl` (228 rows), both merged into `data/raw/LeetCodeDataset.jsonl` (2869 records,
field `hf_split` keeps the original split; the original train/test split is **ignored**, we split ourselves).
Every stage's report stores the `dataset_commit` it was built from; `rhg.data.build` prints
`WARNING: dataset revision changed between stages` if `DATASET_REVISION` differs from the upstream stage's
commit, and `fetch` warns when a re-fetch changes the commit. The fixture uses `rhg-fixture@<hash of content>`.

## 2. Real schema (inspected before writing the converter)

| field | type | used as | notes / quirks |
|---|---|---|---|
| `task_id` | str (LeetCode slug) | `problem_id` | unique across all 2869 records |
| `question_id` | int | – | unused |
| `difficulty` | Easy/Medium/Hard | `difficulty` | 686 / 1498 / 685 raw |
| `tags` | list[str] | `tags` | |
| `problem_description` | str | `description` | contains `\xa0` (NBSP) in all records, `Example 1:` in 2868 and `Constraints:` in all; exponents are flattened (`10^4` is written `104`, `-10^9` as `-109`). Cleaned: NBSP -> space, CRLF -> LF, trailing blanks per line stripped, 3+ newlines collapsed |
| `starter_code` | str | `starter_code` | `class Solution:\n    def f(...):\n        ` (trailing 8 spaces); 174 records start with a `# Definition for ...` comment (128 binary-tree, 46 linked-list) |
| `estimated_date` | timestamp | `date` (`YYYY-MM-DD`) | 2015-08-07 .. 2025-03-30 (in Qwen's pretraining window: contamination is not fixable, see DESIGN §2.2) |
| `prompt` | str | `import_prefix` | **not a prompt**: the import prefix (`import math`, `from typing import *`, ..., `ListNode`/`TreeNode`/`list_node`/`tree_node`/`is_same_list`/`is_same_tree` helpers). Only 4 distinct values; 90 records (2766 use the plain one) also `from sortedcontainers import SortedList` |
| `completion` | str | `reference_solution` | class-based, mostly without imports (34 records contain their own) |
| `entry_point` | str | `entry_point` | always `Solution().<method>` (all 2869); the grader `eval`s it |
| `test` | str | source of `reward_tests`/`heldout_tests` | one `def check(candidate):` per record, body = flat `assert` statements (280,006 in total): `candidate(kw = ...) == value` in 274,914 and `is_same_list/is_same_tree(...)` calls in 5,092. Parses with `ast` in 2869/2869 records (some emit `SyntaxWarning: invalid escape sequence`, silenced) |
| `input_output` | list[{input, output}] | fallback only | strings such as `nums = [3,3], target = 6` / `[0, 1]`; cannot express `list_node(...)`/`tree_node(...)` arguments |
| `query`, `response` | str | unused | LLM prompt/answer used by the dataset authors for SFT |

Decision on tests source: **`test` (parsed with `ast`) for all 2869 records**; `input_output` is only
a fallback when `test` cannot be parsed or has no assert (never triggered on the real data:
`tests_source = {"check": 2869}`) and is refused for problems whose starter code mentions
`ListNode`/`TreeNode`/`Node`. Reason: `test` is the dataset's executable ground truth, it carries the same
cases plus helper-typed ones, and per-assert source text can be executed unchanged
(`candidate` is the entry point).

Statistics of the parse: every `check` is flat (no preamble, no non-assert statements, no compound
statements: `n_problems_with_preamble = 0`, `n_compound_tests = 0`; the parser supports them anyway and
tests it on hand-written `check` functions). **5,577 duplicate asserts were removed** (the dataset repeats
identical cases inside one `check`). After dedupe, tests per problem: min 10 (after the drop rule), median 96,
max 450.

## 3. Conversion, split and drop rules (`rhg.data.load`, `rhg.data.tests_split`)

* `reward_tests` (K = `data.k_reward_tests` = 5) and `heldout_tests` (<= `data.max_heldout_tests` = 20) are
  the first K / next 20 tests in the order `sha256("rhg-test-split-v1|<problem_id>|<test index>")`; the
  function takes no seed, so the split is identical for every training seed and arm. Test `id` = index of
  the assert in the deduplicated list. 2672 of the 2716 final candidates have the full 20 held-out tests, the
  other 44 have 7..19 (always >= 5).
* Drop reasons (tests stage, 2869 -> 2750):

| reason | n | example |
|---|---|---|
| `too_few_tests` (< K+5 = 10 unique asserts) | 34 | `first-bad-version` (1), `clone-binary-tree-with-random-pointer` (1), `count-good-numbers` (3), `generate-parentheses` (8), `check-if-move-is-legal` (8); by test count: 1:5, 2:5, 3:8, 4:2, 5:2, 6:2, 7:1, 8:6, 9:3 |
| `prefix_import_unavailable` | 85 | `add-two-numbers`, `avoid-flood-in-the-city`: the prefix does `from sortedcontainers import SortedList`, and the sandbox only has the standard library |
| `unsupported_helper` (tests use names neither the prefix nor builtins define) | 0 | the prefix supplies `list_node`, `tree_node`, `is_same_list`, `is_same_tree`, so all problems whose tests use them (130 `tree_node`, 45 `list_node`, 32 `is_same_list`, 30 `is_same_tree`, counted per problem) are supported |
| `test_parse_error`, `missing_field` | 0 | |

  Order of checks: too-few first, then prefix, then helper names. 90 records import `sortedcontainers`;
  5 of them are already counted under `too_few_tests`. Decision: drop rather than add the pure-Python package
  to the sandbox/GPU-box dependencies (3% of the data; these are mostly hard problems). Helper-name check =
  free names of all asserts (one `ast` pass) minus names bound by executing the prefix in the sandbox minus builtins;
  execution of the reference (next stage) is the authoritative test.
* Reference validation (validate stage, 2750 -> 2716): first every reference is executed against **all**
  of its asserts in one sandbox process (60 s wall budget per problem, each assert timed), then once more
  through the real grader (`grade_batch`, `sandbox.timeout_s` = 6 s, cache off) on the selected reward + held-out
  tests, with one sequential retry so a loaded machine does not cause spurious drops. Wall time for the
  whole stage: **695.8 s** with 14 workers on Windows (only the wall timeout is enforced there).

| reason | n | detail |
|---|---|---|
| `reference_timeout` (all tests > 60 s) | 31 | genuinely heavy brute-force references, e.g. `coin-change`, `flip-game-ii`, `stickers-to-spell-word`, `unique-paths-iii`, `minimum-knight-moves`; slowest single assert seen: 51.0 s (p99 of per-problem slowest assert: 6.1 s); 71 problems have an assert slower than 1 s |
| `reference_fails_in_grader` (passes visible, times out on held-out in 6 s) | 2 | `sum-of-square-numbers`, `tiling-a-rectangle-with-the-fewest-squares` |
| `reference_crash` | 1 | `zero-array-transformation-iii`: the *reference itself* imports `sortedcontainers` although its prefix does not |
| `reference_fails_tests` (wrong answer) | 0 | |

  **Reference validity = 2716 / 2750 = 98.76 %** (Gate 1c needs >= 95 %); measured against all 2869 records
  it would be 94.7 %, but 119 of those 153 drops are structural (made before validation), not bad references.
  The timeout drops are conservative (a longer `--ref-timeout` would keep some); they depend mildly on machine
  speed, so rebuilding on another box can change the candidate list by a handful of borderline problems, hence
  the final `problems.jsonl`/`split_hash` should be built once and copied to the GPU box rather than rebuilt.
* Final candidates (`data/processed/candidates.jsonl`): **2716** problems, Easy 663 / Medium 1424 / Hard 629.
* Degenerate outputs: 59 of the 2750 converted problems have >= 90 % identical expected values across their
  asserts (text-based approximation, mostly yes/no problems): a constant answer passes most of their tests.
  Not a drop rule (nothing in the pre-registration allows one); such problems are also mostly removed by the
  `p_A` band if the model does not solve them, and remain visible via `VISIBLE_ONLY`.
* Known limitations: some references/tests accept only one of several valid answers (e.g. "return in any
  order" with `==`), so honest solutions can fail; exponents are flattened in the descriptions.

## 4. Near-duplicate clusters (`rhg.data.dedupe`)

Rule (as specified): normalised text (lower-case, cut at the `Example`/`Constraints`/`Follow-up` line,
letters only) -> 5-word shingles, Jaccard >= 0.8 (exact rational comparison); OR identical normalised titles
(slug minus number and Roman-numeral tokens `ii..ix`). Connected components; `cluster_id` = smallest
`problem_id` of the component; clustering runs on all 2869 raw records so ids do not depend on later drops.

Real data (48 Jaccard edges, 149 title edges):

| population | clusters | size histogram (size: clusters) | problems in multi-member clusters | largest |
|---|---|---|---|---|
| all 2869 records | 2673 | 1: 2526, 2: 118, 3: 16, 4: 11, 7: 1, 8: 1 | 343 | 8 |
| 2750 after `tests` | 2574 | 1: 2437, 2: 113, 3: 16, 4: 6, 7: 1, 8: 1 | 313 | 8 |
| 2716 candidates | 2546 | 1: 2413, 2: 111, 3: 14, 4: 6, 7: 1, 8: 1 | 303 | 8 |

Largest clusters are the numbered families (`stone-game` I-IX, `jump-game` I-VIII, `basic-calculator`,
`best-time-to-buy-and-sell-stock`); typical pairs are `contains-duplicate` / `-ii`, `reverse-string` /
`-ii`. The rule is deliberately conservative: sequels that are genuinely different problems are kept
together as well. The `split` stage assigns whole clusters (the band selection is applied first, so only
selected members count).

## 5. Pass-rate file schema (owned by the dataset pipeline; produced by the pass-rate stage on the GPU box)

`data/processed/passrate_A.jsonl` and `passrate_B.jsonl`, one JSON object per line:

```json
{"problem_id": "two-sum", "n": 16, "k_visible": 3, "k_full": 2}
```

`n` samples were drawn (may differ per stage/problem), `k_visible` passed the reward (visible) tests, `k_full`
passed reward + held-out tests, so `0 <= k_full <= k_visible <= n`, `n > 0`. Stage A covers all candidates
(`candidates.jsonl`); stage B only the band-selected problems, from *independent* samples. Unknown ids,
duplicates, out-of-range counts and missing stage-B rows are errors (exit code 2).
`p_A = k_visible/n` (stage A), `p_B_visible = k_visible/n` and `p_B_full = k_full/n` (stage B).
`python -m rhg.data.build --stage split --select-only [--widen]` writes `selected_A.json` (band + selected ids) so
stage B knows what to sample.

## 6. The `split` stage

* Band: `data.band_low <= p_A <= data.band_high` (0.10 / 0.40 inclusive; exact `Fraction` arithmetic).
  `--widen` replaces it with the single pre-declared fallback `[0.05, 0.50]` (also inclusive) and records
  `band.widened = true` in `splits.json`; the CLI banner then says WIDENED. Nothing else can change the band.
* Strata: terciles of the **cluster-mean** `p_A` (cut on cumulative problem counts, ties broken by a hash).
* Sizes: `n_test = min(60, round(0.24 N))`, `n_val = min(40, round(0.16 N))`, rest train (N = selected
  problems). With N >= 250 this is exactly 60/40 (the affordable evaluation sizes, DESIGN §6: >= 480 test
  rollouts per seed); below that the same ratios apply and Gate 1c reports the shortfall.
* Assignment: per stratum, clusters in a seeded pseudo-random order (`SPLIT_SEED = "rhg-split-v1"`, a fixed
  constant, independent of the training seed) go to test/val/train by largest relative deficit when they fit;
  a fix-up pass moves the smallest train clusters into val/test until both reach their targets, so a cluster is
  never split and val/test are never smaller than targeted while avoidable.
* Outputs: `problems.jsonl` (REPO_SPEC §5 schema incl. `cluster_id, p_A, p_B_full, p_B_visible, split`) and
  `splits.json` (`split_hash` = sha256 of the sorted `"<id>:<split>"` lines joined by `\n`, lists per split, counts, band,
  `dataset_revision` line, balance statistics, `gate1c`).
* Gate-1c checklist (printed; `--strict` makes a failure exit 1; default exit code is 0 and the outcome is in
  `splits.json["gate1c"]`): train >= 150, val >= 40, test >= 60, reference validity >= 95 % (from
  `validation_report.json`; missing report = FAIL), `p_A` balance (Kruskal-Wallis p >= 0.10), difficulty mix
  (chi-square p >= 0.05, informational WARN only), no cluster spans splits. Balance thresholds are chosen here
  and are heuristics, not pre-registered values.

## 7. Prompts and chat rendering (`rhg.data.prompts`)

Template and hints come from `configs/prompts.yaml`; `build_prompt` fills it with a single regex pass (so braces in
problem text are safe) and strips the result; the hint is the last paragraph and `none` adds nothing. Tests
assert `prompt(hint) == prompt(none) + "\n\n" + wording` for every level and wording id, and
`strip_hint(prompt(hint)) == prompt(none)` for all wordings (S1-S3, E1; E1 contains S1 and is removed first).
`prompts_hash(cfg)` = sha256 of the canonical JSON of template + hints + `subtle_selected` (the `frozen` flag is
excluded); `prompts_hash(path)` delegates to the manifest's file hash.

**Tokenizer check (run once on 2026-09-19, `pytest -m network`)**: the fallback renderer
`render_chat(text)` equals the real Qwen3-1.7B chat template output, with thinking off
(`...<|im_start|>assistant\n<think>\n\n</think>\n\n`) and on, via (a) `jinja2` rendering of the downloaded
`tokenizer_config.json` template and (b) `transformers 5.17.0` / `tokenizers 0.23.2`
`AutoTokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=...)`. Only tokenizer files were
downloaded (no weights). `transformers` is a GPU-stack package, not a core dependency: route (b) is skipped
unless installed (`uv run --with transformers pytest -m network tests/test_prompts.py`).

## 8. Fixture

`rhg.data.fixture` emits 40 synthetic problems in the raw dataset schema (so the real conversion code runs on
them): 15 Easy / 17 Medium / 8 Hard (including the 3 planted variants), `Solution().f` entry points for most and plain
functions for 3, >= 25 asserts each (>= 15 required), 4 degenerate-output problems (>= 80 % same expected value),
3 slow-only held-out cases (huge input placed in a held-out slot for K=5/20; naive solutions in
`NAIVE_SLOW_SOLUTIONS` time out on it while passing the reward tests) and 3 planted near-duplicate pairs
(numbers-only variant, one-word paraphrase, `-ii` title variant). `tests/fixtures/problems_tiny.jsonl` is the
committed 8-problem copy with the full `problems.jsonl` schema (synthetic `p_A`/`p_B_*`/`split`); a test checks it
matches `fixture.tiny_problems()`. `--fixture` on the CLI works under `data/fixture/{raw,processed}` and never
touches real data; its `split` stage fabricates deterministic pass-rate files when none exist. With only 40
problems Gate 1c naturally FAILS there (that is tested).

## 9. Reproduce

```
uv run python -m rhg.data.build --stage fetch        # data/raw + DATASET_REVISION (needs network)
uv run python -m rhg.data.build --stage tests
uv run python -m rhg.data.build --stage validate     # ~12 min on 14 workers, Windows
uv run python -m rhg.data.dedupe                     # cluster histogram of converted.jsonl
# after `rhg.eval.pass_rate --stage A` on the GPU box:
uv run python -m rhg.data.build --stage split --select-only [--widen]
# after stage B:
uv run python -m rhg.data.build --stage split [--widen] [--strict]
```
