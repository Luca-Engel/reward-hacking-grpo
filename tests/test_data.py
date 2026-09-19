"""Dataset pipeline: check() parsing, test splitting, drops, dedupe, band/split stage (subtask 04).

Everything runs on the synthetic fixture or hand-written snippets (no network). Statements executed in the
sandbox are the fixture's own trusted reference solutions and tiny hand-written programs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import warnings
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pytest

from rhg.config import load_config
from rhg.data import build, dedupe, fixture, load
from rhg.data import tests_split as ts

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


# ====================================================================== check() parsing
def run_check_source(check_src: str, candidate) -> bool:
    ns: dict = {}
    exec(compile(check_src, "<check>", "exec"), ns)
    try:
        ns["check"](candidate)
        return True
    except AssertionError:
        return False


def run_tests_on(tests: list[str], candidate) -> bool:
    for src in tests:
        try:
            exec(compile(src, "<t>", "exec"), {"candidate": candidate})
        except AssertionError:
            return False
    return True


CHECK_BASIC = """def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(a = 5, b = 5) == 10
    assert candidate(0, 0) == 0
"""


def test_parse_multiple_asserts():
    parsed = ts.parse_check(CHECK_BASIC)
    assert parsed.source == "check"
    assert parsed.tests == [
        "assert candidate(1, 2) == 3",
        "assert candidate(a = 5, b = 5) == 10",
        "assert candidate(0, 0) == 0",
    ]
    assert parsed.n_raw_asserts == 3 and parsed.preamble == ""


def test_parse_is_equivalent_to_running_check():
    """check(candidate) passes iff every parsed test passes (checked on good and bad candidates)."""
    good = lambda a, b: a + b
    bad_one = lambda a, b: a + b + (1 if (a, b) == (5, 5) else 0)
    bad_all = lambda a, b: 0
    parsed = ts.parse_check(CHECK_BASIC)
    for cand in (good, bad_one, bad_all):
        assert run_tests_on(parsed.tests, cand) == run_check_source(CHECK_BASIC, cand)
    assert run_tests_on(parsed.tests, good) and not run_tests_on(parsed.tests, bad_one)


CHECK_PREAMBLE = '''import math
HELPER = 10

def check(candidate):
    """docstring is not a test"""
    OFFSET = 3
    assert candidate(1) == 1 + OFFSET
    pass
    assert candidate(2) == math.floor(2.5) + OFFSET
    OFFSET = 100
    assert candidate(3) == 3 + OFFSET + HELPER - 10
    print("non-assert statement")
    return None
'''


def test_parse_preamble_and_non_assert_statements():
    parsed = ts.parse_check(CHECK_PREAMBLE)
    assert len(parsed.tests) == 3 and "preamble" in parsed.notes
    # module-level imports/assignments and in-function assignments prepend to every later test
    assert all("import math" in t and "HELPER = 10" in t for t in parsed.tests)
    assert "OFFSET = 3" in parsed.tests[0] and "OFFSET = 100" not in parsed.tests[0]
    assert "OFFSET = 100" in parsed.tests[2]
    assert not any("pass" in t.split("\n")[-1] or "docstring" in t for t in parsed.tests)
    assert run_tests_on(parsed.tests, lambda x: {1: 4, 2: 5, 3: 103}[x])
    assert not run_tests_on(parsed.tests, lambda x: 4)
    assert run_check_source(CHECK_PREAMBLE, lambda x: {1: 4, 2: 5, 3: 103}[x])


def test_parse_compound_duplicates_and_multiline():
    code = '''def check(candidate):
    assert candidate("a\\nb") == 2
    assert candidate("a\\nb") == 2
    for i in range(3):
        assert candidate(str(i)) == 1
    assert candidate("""x
y""") == 2
'''
    parsed = ts.parse_check(code)
    assert parsed.n_raw_asserts == 4 and parsed.n_duplicates == 1 and parsed.n_compound == 1
    assert len(parsed.tests) == 3
    assert parsed.tests[1].startswith("for i in range(3):")
    assert parsed.tests[2] == 'assert candidate("""x\ny""") == 2'  # multi-line literal kept verbatim
    cand = lambda s: len(s.split("\n")) if len(s) > 1 else 1
    assert run_tests_on(parsed.tests, cand) == run_check_source(code, cand)


def test_parse_errors_and_warnings():
    with pytest.raises(ts.CheckParseError):
        ts.parse_check("def check(candidate:\n  assert 1")
    with pytest.raises(ts.CheckParseError):
        ts.parse_check("def other(candidate):\n    assert 1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # invalid escape sequences in dataset strings must not surface
        parsed = ts.parse_check("def check(candidate):\n    assert candidate('\\:') == 1\n    assert candidate('\\p') == 2\n")
    assert len(parsed.tests) == 2
    assert ts.parse_check("def check(candidate):\n    return\n").tests == []


def test_input_output_fallback():
    pairs = [
        {"input": "nums = [3,3], target = 6", "output": "[0, 1]"},
        {"input": "nums = [1], target = 2", "output": "None"},
        {"input": "nums = [3,3], target = 6", "output": "[0, 1]"},  # duplicate
        {"input": "nums = [", "output": "broken"},  # unparseable
    ]
    parsed = ts.tests_from_input_output(pairs)
    assert parsed.source == "input_output"
    assert parsed.tests == ["assert candidate(nums = [3,3], target = 6) == [0, 1]", "assert candidate(nums = [1], target = 2) == None"]
    assert parsed.n_duplicates == 1 and any("unparseable" in n for n in parsed.notes)


def raw_record(pid="t-problem", n_asserts=12, prefix=fixture.FIXTURE_PREFIX, test=None, **over):
    body = "".join(f"    assert candidate(x = {i}) == {i + 1}\n" for i in range(n_asserts))
    rec = {
        "task_id": pid, "question_id": 1, "difficulty": "Easy", "tags": ["Array"],
        "problem_description": "Add one to x.\n\nExample 1:\n\nInput: x = 1\nOutput: 2\n\nConstraints:\n\n1 <= x <= 5",
        "starter_code": "class Solution:\n    def inc(self, x: int) -> int:\n        ", "estimated_date": "2020-01-02T00:00:00",
        "prompt": prefix, "completion": "class Solution:\n    def inc(self, x: int) -> int:\n        return x + 1\n",
        "entry_point": "Solution().inc", "test": test if test is not None else "def check(candidate):\n" + body,
        "input_output": [{"input": f"x = {i}", "output": str(i + 1)} for i in range(n_asserts)],
    }
    rec.update(over)
    return rec


# ====================================================================== test splitting
def brute_force_order(problem_id: str, n: int) -> list[int]:
    keyed = []
    for i in range(n):
        digest = hashlib.sha256(f"rhg-test-split-v1|{problem_id}|{i}".encode("utf-8")).digest()
        keyed.append((digest, i))
    return [i for _, i in sorted(keyed)]


def test_split_tests_matches_documented_hash_and_is_disjoint():
    tests = [f"assert candidate({i}) == {i}" for i in range(60)]
    for pid in ("alpha", "beta", "two-sum"):
        reward, held = ts.split_tests(pid, tests, 5, 20)
        order = brute_force_order(pid, 60)
        assert [t["id"] for t in reward] == sorted(order[:5])
        assert [t["id"] for t in held] == sorted(order[5:25])
        assert not {t["id"] for t in reward} & {t["id"] for t in held}
        assert len(reward) == 5 and len(held) == 20
        assert all(t["kind"] == "assert" and t["src"] == tests[t["id"]] for t in reward + held)
    r1, _ = ts.split_tests("alpha", tests, 5, 20)
    r2, _ = ts.split_tests("beta", tests, 5, 20)
    assert [t["id"] for t in r1] != [t["id"] for t in r2]  # depends on the problem id


def test_split_respects_k_and_max_heldout_and_min_tests():
    tests = [f"assert candidate({i}) == {i}" for i in range(12)]
    for k in (1, 5, 7):
        reward, held = ts.split_tests("p", tests, k, 20)
        assert len(reward) == k and len(held) == min(20, 12 - k) and len(held) >= 5
    reward, held = ts.split_tests("p", tests, 5, 3)
    assert (len(reward), len(held)) == (5, 3)
    with pytest.raises(ValueError):
        ts.split_tests("p", tests[:9], 5, 20)  # < K+5
    assert len(ts.split_tests("p", tests[:10], 5, 20)[1]) == 5


def test_split_is_independent_of_training_seed():
    """The split is a function of (problem_id, index) only: no seed enters, whatever cfg/rng state is."""
    base = fixture.fixture_raw_records()[:6]
    results = []
    for seed in (0, 1, 4242):
        cfg = load_config("clean_none", seed=seed)
        random.seed(seed)
        problems, _, _ = load.convert_records(base, cfg.data.k_reward_tests, cfg.data.max_heldout_tests, workers=1)
        results.append([(p["problem_id"], [t["id"] for t in p["reward_tests"]], [t["id"] for t in p["heldout_tests"]]) for p in problems])
    assert results[0] == results[1] == results[2]
    import inspect

    assert "seed" not in inspect.signature(ts.split_tests).parameters


def test_fixture_problems_respect_k_and_disjointness():
    for k, mh in ((5, 20), (3, 10)):
        for p in fixture.fixture_candidates(k, mh):
            r, h = p["reward_tests"], p["heldout_tests"]
            assert len(r) == k and 5 <= len(h) <= mh
            assert not {t["id"] for t in r} & {t["id"] for t in h}
            assert len({t["id"] for t in r + h}) == len(r) + len(h)


def stub_names(prefix: str):
    return {"list_node"} if "def list_node" in prefix else set()


@pytest.mark.parametrize(
    "n_asserts,expected",
    [(9, load.DROP_TOO_FEW), (10, None), (25, None)],
)
def test_drop_too_few_tests_boundary(n_asserts, expected):
    conv = load.convert_record(raw_record(n_asserts=n_asserts), 5, 20, prefix_names=stub_names)
    assert conv.drop_reason == expected
    if expected is None:
        assert len(conv.problem["reward_tests"]) == 5 and len(conv.problem["heldout_tests"]) == min(20, n_asserts - 5)


def test_drop_unsupported_helper_types_using_the_real_prefix_check():
    body = "def check(candidate):\n" + "".join(f"    assert is_same_list(candidate(l1 = list_node([{i}])), list_node([{i}]))\n" for i in range(12))
    bare = "import math\n"
    with_helper = bare + "def list_node(v):\n    return v\ndef is_same_list(a, b):\n    return a == b\n"
    conv = load.convert_record(raw_record(test=body, prefix=bare), 5, 20)  # runs the prefix in the sandbox
    assert conv.drop_reason == load.DROP_HELPER and "list_node" in conv.detail and "is_same_list" in conv.detail
    ok = load.convert_record(raw_record(test=body, prefix=with_helper), 5, 20)
    assert ok.drop_reason is None and ok.problem is not None


def test_drop_prefix_import_unavailable_and_missing_fields_and_parse_error():
    conv = load.convert_record(raw_record(prefix="import definitely_not_a_module_xyz\n"), 5, 20)
    assert conv.drop_reason == load.DROP_PREFIX and "definitely_not_a_module_xyz" in conv.detail
    assert load.convert_record(raw_record(completion=""), 5, 20).drop_reason == load.DROP_MISSING_FIELD
    # unparseable `test` -> input_output fallback for a plain function...
    fb = load.convert_record(raw_record(test="def check(candidate:\n"), 5, 20, prefix_names=stub_names)
    assert fb.problem is not None and fb.tests_source == "input_output"
    # ...but not for ListNode problems (cannot express the helper values)
    ln = load.convert_record(raw_record(test="def check(candidate:\n", starter_code="class Solution:\n    def f(self, head: Optional[ListNode]):\n        "), 5, 20, prefix_names=stub_names)
    assert ln.drop_reason == load.DROP_PARSE_ERROR


def test_free_names():
    assert ts.free_names("assert candidate(x = [1, 2]) == 3") == set()
    assert ts.free_names("assert candidate(head = list_node([1])) == tree_node([2])") == {"list_node", "tree_node"}
    assert ts.free_names("assert sorted(candidate(a = [i for i in range(3)])) == [len(q) for q in ['a']]") == set()
    assert ts.free_names("import math\nassert candidate(1) == math.pi") == set()
    assert ts.free_names("assert (lambda z: z + w)(1) == candidate(2)") == {"w"}


def test_convert_conversion_details():
    conv = load.convert_record(raw_record(pid="x", problem_description="A\xa0b\r\nline  \n\n\n\nEnd"), 5, 20, prefix_names=stub_names)
    p = conv.problem
    assert p["description"] == "A b\nline\n\nEnd"
    assert p["problem_id"] == "x" and p["source"] == load.SOURCE and p["date"] == "2020-01-02"
    assert p["entry_point"] == "Solution().inc" and p["reference_solution"].startswith("class Solution")
    assert p["import_prefix"] == fixture.FIXTURE_PREFIX and p["difficulty"] == "Easy" and p["tags"] == ["Array"]
    assert set(p) == {"problem_id", "source", "difficulty", "tags", "date", "description", "starter_code", "import_prefix",
                      "entry_point", "reference_solution", "reward_tests", "heldout_tests"}


def test_convert_records_pool_matches_sequential():
    base = fixture.fixture_raw_records()
    records = []
    for i in range(2):
        for r in base:
            r = dict(r)
            r["task_id"] = f"{r['task_id']}-copy{i}"
            r["problem_description"] = f"Copy {i} of a problem with a uniquely worded statement {r['task_id']}."
            records.append(r)
    assert len(records) >= 64
    seq = load.convert_records(records, 5, 20, workers=1)
    par = load.convert_records(records, 5, 20, workers=2)
    assert seq[0] == par[0] and seq[1] == par[1]
    assert len(seq[0]) == len(records)


# ====================================================================== fixture
def test_fixture_shape_and_determinism():
    raw = fixture.fixture_raw_records()
    assert 38 <= len(raw) <= 44
    assert raw == fixture.fixture_raw_records()
    assert fixture.fixture_revision() == fixture.fixture_revision()
    assert {r["difficulty"] for r in raw} == {"Easy", "Medium", "Hard"}
    assert len({r["task_id"] for r in raw}) == len(raw)
    for r in raw:
        parsed = ts.parse_check(r["test"])
        assert len(parsed.tests) >= 15, r["task_id"]
        assert r["entry_point"] in ("Solution()." + r["completion"].split("def ")[1].split("(")[0], r["completion"].split("def ")[1].split("(")[0])
    assert any(r["entry_point"].startswith("Solution()") for r in raw) and any("(" not in r["entry_point"] for r in raw)


def test_fixture_degenerate_problems_have_constant_expected_output():
    by_id = {r["task_id"]: r for r in fixture.fixture_raw_records()}
    assert set(fixture.DEGENERATE_IDS) <= set(by_id)
    for pid in fixture.DEGENERATE_IDS:
        rhs = Counter(t.rsplit(" == ", 1)[1] for t in ts.parse_check(by_id[pid]["test"]).tests)
        assert rhs.most_common(1)[0][1] / sum(rhs.values()) >= 0.8, pid
    other = Counter(t.rsplit(" == ", 1)[1] for t in ts.parse_check(by_id["digit-sum"]["test"]).tests)
    assert other.most_common(1)[0][1] / sum(other.values()) < 0.3


def test_fixture_slow_heldout_case_lands_in_heldout_and_times_out_naive_code():
    from rhg.env.grader import grade_batch, GradeItem

    cands = {p["problem_id"]: p for p in fixture.fixture_candidates()}
    cfg = load_config("clean_none", overrides=["sandbox.timeout_s=2.5", "sandbox.cache=false"])
    items = []
    for pid in fixture.SLOW_HELDOUT_IDS:
        p = cands[pid]
        big = [t for t in p["heldout_tests"] if re.search(r"\d{11,}", t["src"])]
        assert len(big) == 1, pid  # exactly one huge test, and it is a held-out one
        assert not any(re.search(r"\d{11,}", t["src"]) for t in p["reward_tests"])
        items.append(GradeItem(p, "```python\n" + fixture.NAIVE_SLOW_SOLUTIONS[pid] + "\n```", "clean", cfg=cfg))
    for res in grade_batch(items, workers=3, cfg=cfg):
        assert res.raw["visible_pass"] is True  # correct on the reward tests...
        assert res.raw["heldout_pass"] is False and res.raw["timeout"] is True  # ...times out on the slow held-out one
        assert res.reward == 1.0  # the reward never sees the held-out timeout


def test_tiny_fixture_copy_is_current():
    committed = [json.loads(line) for line in (FIXTURES / "problems_tiny.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
    assert committed == json.loads(json.dumps(fixture.tiny_problems()))
    assert len(committed) == 8
    spec_keys = {"problem_id", "source", "difficulty", "tags", "date", "description", "starter_code", "import_prefix", "entry_point",
                 "reference_solution", "reward_tests", "heldout_tests", "cluster_id", "p_A", "p_B_full", "p_B_visible", "split"}
    assert all(set(p) == spec_keys for p in committed)
    assert all(0.10 <= p["p_A"] <= 0.40 for p in committed)
    assert Counter(p["split"] for p in committed) == {"train": 4, "val": 2, "test": 2}


# ====================================================================== dedupe
def brute_force_words(text: str) -> list[str]:
    """Independent re-implementation of the documented normalisation (no regex reuse from the module)."""
    text = text.replace("\xa0", " ").lower()
    for marker in ("\nexample", "\nconstraints", "\nfollow"):
        pos = text.find(marker)
        if pos != -1:
            text = text[:pos]
    letters = "".join(ch if "a" <= ch <= "z" else " " for ch in text)
    return letters.split()


def brute_force_jaccard(a: str, b: str) -> Fraction:
    def sh(words):
        return {tuple(words[i:i + 5]) for i in range(len(words) - 4)} if len(words) >= 5 else {tuple(words)}

    sa, sb = sh(brute_force_words(a)), sh(brute_force_words(b))
    return Fraction(len(sa & sb), len(sa | sb))


LONG = (
    "Given an array of integers you must find the longest stretch of consecutive positions whose values keep "
    "strictly increasing and then return the length of that stretch. The array can be empty in which case the "
    "answer is zero. You should aim for a solution that reads the array once from left to right and keeps only "
    "a couple of counters while doing so, because the input can be large."
)


def prob(pid, text):
    return {"problem_id": pid, "description": text}


def test_planted_pairs_cluster_and_distinct_problems_do_not():
    numbers = LONG + "\n\nExample 1:\n\nInput: nums = [1,2]\nOutput: 2\n\nConstraints:\n\n1 <= nums.length <= 10"
    numbers2 = LONG + "\n\nExample 1:\n\nInput: nums = [7,8,9]\nOutput: 3\n\nConstraints:\n\n1 <= nums.length <= 5000"
    paraphrase = LONG.replace("stretch of consecutive", "run of consecutive", 1)
    other = ("Design a data structure that supports inserting a value and retrieving a random stored value, each in "
             "constant average time, where every currently stored value must be equally likely to be returned by a call.")
    assert brute_force_jaccard(numbers, numbers2) == 1
    assert brute_force_jaccard(LONG, paraphrase) >= Fraction(4, 5)
    assert brute_force_jaccard(LONG, other) < Fraction(1, 5)
    a = dedupe.cluster_problems([prob("aaa-one", numbers), prob("bbb-two", numbers2), prob("ccc-three", paraphrase), prob("ddd-other", other)])
    assert a["aaa-one"] == a["bbb-two"] == a["ccc-three"] == "aaa-one"
    assert a["ddd-other"] == "ddd-other"


def test_fixture_clusters_are_exactly_the_planted_ones():
    cands = fixture.fixture_candidates()
    groups = defaultdict(set)
    for p in cands:
        groups[p["cluster_id"]].add(p["problem_id"])
    multi = sorted(sorted(g) for g in groups.values() if len(g) > 1)
    assert multi == sorted(sorted(pair) for pair in fixture.PLANTED_CLUSTERS)
    # brute force over every pair with the independent implementation
    texts = {r["task_id"]: r["problem_description"] for r in fixture.fixture_raw_records()}
    linked = {frozenset((a, b)) for a in texts for b in texts if a < b and brute_force_jaccard(texts[a], texts[b]) >= Fraction(4, 5)}
    title_linked = {frozenset(pair) for pair in fixture.PLANTED_CLUSTERS if dedupe.normalize_title(pair[0]) == dedupe.normalize_title(pair[1])}
    expected = linked | title_linked
    assert expected == {frozenset(pair) for pair in fixture.PLANTED_CLUSTERS}


def test_jaccard_threshold_is_exact_and_inclusive():
    eight = "alpha beta gamma delta epsilon zeta eta theta"  # 4 shingles
    nine = eight + " iota"  # 5 shingles, the 4 above included: Jaccard = 4/5 exactly
    assert brute_force_jaccard(eight, nine) == Fraction(4, 5)
    ps = [prob("p-one", eight), prob("q-two", nine)]
    assert dedupe.cluster_problems(ps, "0.8")["q-two"] == "p-one"
    assert dedupe.cluster_problems(ps, "0.81")["q-two"] == "q-two"
    assert dedupe.cluster_problems(ps, 0.8)["q-two"] == "p-one"


def test_near_miss_below_threshold_stays_apart():
    words = LONG.split()
    changed = list(words)
    for i in range(5, len(words), 9):  # every 9th word changed -> most shingles differ
        changed[i] = "zzzz"
    b = " ".join(changed)
    j = brute_force_jaccard(LONG, b)
    assert j < Fraction(4, 5)
    a = dedupe.cluster_problems([prob("x-one", LONG), prob("y-two", b)])
    assert a["x-one"] != a["y-two"]


def test_title_rule_and_normalisation():
    assert dedupe.normalize_title("Two-Sum-II") == dedupe.normalize_title("two-sum") == "two-sum"
    assert dedupe.normalize_title("game-of-life-2") == "game-of-life"
    assert dedupe.normalize_title("two-sum-ii-input-array-is-sorted") == "two-sum-input-array-is-sorted"
    ps = [prob("merge-things-ii", "completely different words about trees and forests and lakes today okay"),
          prob("merge-things", "an unrelated sentence regarding matrices vectors kernels and their spectra now"),
          prob("merge-other", "a third story about graphs and cycles and shortest routes between cities here")]
    a = dedupe.cluster_problems(ps, return_stats=True)
    assign, stats = a
    assert assign["merge-things-ii"] == assign["merge-things"] == "merge-things"
    assert assign["merge-other"] == "merge-other"
    assert stats == {"edges_jaccard": 0, "edges_title": 1}


def test_normalisation_strips_examples_numbers_and_punctuation():
    text = "Find  the MAX, of 12 numbers!\n\nExample 1:\n\nInput: [1,2]\nOutput: 2\n\nConstraints:\n\n1 <= n <= 10^5"
    assert dedupe.normalize_text(text) == ["find", "the", "max", "of", "numbers"]
    assert dedupe.normalize_text(text) == brute_force_words(text)
    assert dedupe.shingles(["a", "b", "c"]) == frozenset({"a b c"})  # short texts: one shingle
    assert dedupe.shingles([]) == frozenset()


def test_transitive_chain_forms_one_cluster():
    words = ["".join(pair) for pair in itertools.product("abcdefg", repeat=2)]  # 49 distinct letter-only tokens
    a = " ".join(words[0:24])  # 20 shingles
    b = " ".join(words[2:26])  # shifted by 2: Jaccard 18/22
    c = " ".join(words[4:28])  # b~c the same, but a~c only 16/24
    assert brute_force_jaccard(a, b) >= Fraction(4, 5) and brute_force_jaccard(b, c) >= Fraction(4, 5)
    assert brute_force_jaccard(a, c) < Fraction(4, 5)
    assign = dedupe.cluster_problems([prob("z-c", c), prob("m-b", b), prob("k-a", a)])
    assert set(assign.values()) == {"k-a"}


def test_cluster_id_is_deterministic_and_order_independent():
    cands = fixture.fixture_raw_records()
    ps = [{"problem_id": r["task_id"], "description": r["problem_description"]} for r in cands]
    base = dedupe.cluster_problems(ps)
    for seed in range(4):
        shuffled = list(ps)
        random.Random(seed).shuffle(shuffled)
        assert dedupe.cluster_problems(shuffled) == base
    assert all(cid == min(pid for pid, c in base.items() if c == cid) for cid in set(base.values()))
    with pytest.raises(ValueError):
        dedupe.cluster_problems([prob("a", "x"), prob("a", "y")])


def test_cluster_histogram():
    h = dedupe.cluster_histogram({"a": "a", "b": "a", "c": "a", "d": "d", "e": "e", "f": "e"})
    assert h == {"size_histogram": {1: 1, 2: 1, 3: 1}, "n_clusters": 3, "n_problems": 6, "n_affected": 5, "largest": 3}


def test_dedupe_cli_fixture(capsys):
    assert dedupe.main(["--fixture"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["size_histogram"] == {"1": 34, "2": 3} and out["n_affected"] == 6


# ====================================================================== build stages on the fixture
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixture_build")
    raw, proc = root / "raw", root / "processed"
    common = ["--fixture", "--raw-dir", str(raw), "--processed-dir", str(proc)]
    for stage in ("fetch", "tests", "validate"):
        assert build.main(["--stage", stage, *common]) == 0
    return raw, proc, common


def test_fetch_writes_revision_and_raw(built):
    raw, proc, _ = built
    assert (raw / "fixture_raw.jsonl").is_file()
    line = (proc / "DATASET_REVISION").read_text(encoding="utf-8").strip()
    assert line.startswith(fixture.fixture_revision()) and "datasets=" in line and "downloaded=" in line
    assert load.parse_revision_commit(line) == fixture.fixture_revision().split("@")[1]
    assert load.parse_revision_commit("newfacade/LeetCodeDataset@abc123 datasets=5.0.1 downloaded=2026-09-19") == "abc123"
    assert load.parse_revision_commit("") is None and load.parse_revision_commit(None) is None


def test_tests_and_validate_outputs(built):
    _, proc, _ = built
    conv = build.read_jsonl(proc / "converted.jsonl")
    cands = build.read_jsonl(proc / "candidates.jsonl")
    rep = json.loads((proc / "validation_report.json").read_text(encoding="utf-8"))
    assert len(conv) == len(cands) == 40 and rep["n_valid"] == 40 and rep["reference_validity"] == 1.0
    assert cands == sorted(cands, key=lambda p: p["problem_id"])
    assert all("p_A" not in p and "split" not in p and "cluster_id" in p for p in cands)
    trep = json.loads((proc / "tests_report.json").read_text(encoding="utf-8"))
    assert trep["dataset_commit"] == fixture.fixture_revision().split("@")[1] and trep["n_kept"] == 40
    assert trep["clusters_kept"]["size_histogram"] == {"1": 34, "2": 3}


def test_validate_drops_bad_references(tmp_path):
    raw, proc = tmp_path / "raw", tmp_path / "processed"
    build.ensure_raw(raw, proc, True)
    cfg = load_config("clean_none", overrides=["sandbox.timeout_s=2.0", "sandbox.mem_mb=2048"])
    subset = [p for p in fixture.fixture_candidates() if p["problem_id"] in ("digit-sum", "array-range", "count-vowels", "steps-to-zero")]
    by = {p["problem_id"]: p for p in subset}
    by["digit-sum"]["reference_solution"] = "class Solution:\n    def digitSum(self, num: int) -> int:\n        return num % 10\n"  # wrong
    by["array-range"]["reference_solution"] = "class Solution:\n    def arrayRange(self, nums):\n        while True:\n            pass\n"  # hangs
    by["count-vowels"]["reference_solution"] = "def countVowels(s: str) -> int:\n    return sum(1 for c in s if c in 'aeiou')\n"  # fine
    by["steps-to-zero"]["reference_solution"] = (  # correct but 3 s once: passes stage 1 (6 s), exceeds the 2 s grader timeout
        "import time\n_slept = []\nclass Solution:\n    def numberOfSteps(self, num: int) -> int:\n"
        "        if not _slept:\n            _slept.append(1)\n            time.sleep(3)\n"
        "        steps = 0\n        while num:\n            num = num // 2 if num % 2 == 0 else num - 1\n            steps += 1\n        return steps\n"
    )
    build.write_jsonl(proc / "converted.jsonl", subset)
    rep = build.stage_validate(raw, proc, cfg, True, workers=4, timeout_s=6.0, progress=False)
    drops = {d["problem_id"]: d["reason"] for d in rep["drops"]}
    assert drops == {"digit-sum": "reference_fails_tests", "array-range": "reference_timeout", "steps-to-zero": "reference_fails_in_grader"}
    assert rep["n_valid"] == 1 and rep["reference_validity"] == 0.25
    assert [p["problem_id"] for p in build.read_jsonl(proc / "candidates.jsonl")] == ["count-vowels"]


def test_revision_change_between_stages_warns(built, capsys):
    _, proc, common = built
    rev = proc / "DATASET_REVISION"
    original = rev.read_text(encoding="utf-8")
    try:
        rev.write_text("rhg-fixture@deadbeef datasets=n/a downloaded=fixture\n", encoding="utf-8")
        build.main(["--stage", "split", "--select-only", *common])
        err = capsys.readouterr().err
        assert "WARNING: dataset revision changed between stages" in err and "deadbeef" in err
    finally:
        rev.write_text(original, encoding="utf-8")
    capsys.readouterr()
    build.main(["--stage", "split", "--select-only", *common])
    assert "WARNING" not in capsys.readouterr().err


def test_cli_help_and_usage_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        build.main(["--help"])
    assert e.value.code == 0
    assert "--stage" in capsys.readouterr().out
    assert build.main(["--stage", "tests", "--fixture", "--raw-dir", str(tmp_path / "none"), "--processed-dir", str(tmp_path / "p")]) == 2
    assert build.main(["--stage", "split", "--fixture", "--processed-dir", str(tmp_path / "empty")]) == 2
    assert build.main(["--stage", "tests", "--fixture", "--set", "data.k_reward_tests=abc"]) == 2


# ====================================================================== band selection + split stage
def synthetic_candidates(n: int, cluster_sizes=None, seed: int = 1) -> list[dict]:
    """n minimal problems (only the fields the split stage needs) with the given cluster structure."""
    rng = random.Random(seed)
    ids = [f"p{i:04d}" for i in range(n)]
    clusters: dict[str, str] = {}
    i = 0
    for size in (cluster_sizes or []):
        members = ids[i:i + size]
        for m in members:
            clusters[m] = members[0]
        i += size
    diffs = ["Easy", "Medium", "Hard"]
    return [{"problem_id": pid, "cluster_id": clusters.get(pid, pid), "difficulty": rng.choice(diffs), "source": "synthetic",
             "description": pid, "reward_tests": [], "heldout_tests": []} for pid in ids]


def write_pass_files(proc: Path, cands, a_rows, b_rows=None, validity: float | None = 1.0):
    proc.mkdir(parents=True, exist_ok=True)
    build.write_jsonl(proc / "candidates.jsonl", cands)
    build.write_jsonl(proc / "passrate_A.jsonl", a_rows)
    if b_rows is not None:
        build.write_jsonl(proc / "passrate_B.jsonl", b_rows)
    report = proc / "validation_report.json"
    if validity is None:
        report.unlink(missing_ok=True)
    else:
        report.write_text(json.dumps({"reference_validity": validity}), encoding="utf-8")


def spread_rates(cands, n=20, lo=0, hi=None, seed=2):
    """Stage A/B rows with k_visible spread over [lo, hi] (a full cycle when hi-lo+1 is coprime with 37)."""
    hi = n if hi is None else hi
    rng = random.Random(seed)
    a, b = [], []
    for i, p in enumerate(cands):
        k = lo + (i * 37) % (hi - lo + 1)
        a.append({"problem_id": p["problem_id"], "n": n, "k_visible": k, "k_full": max(0, k - rng.randint(0, 2))})
        kb = min(n, max(0, k + rng.randint(-2, 2)))
        b.append({"problem_id": p["problem_id"], "n": n, "k_visible": kb, "k_full": max(0, kb - rng.randint(0, 3))})
    return a, b


@pytest.mark.parametrize("low,high", [(0.10, 0.40), (0.05, 0.50), (0.0, 1.0)])
def test_in_band_matches_integer_brute_force(low, high):
    lo, hi = Fraction(str(low)), Fraction(str(high))
    for n in range(1, 41):
        for k in range(0, n + 1):
            expect = lo.numerator * n <= k * lo.denominator and k * hi.denominator <= hi.numerator * n
            assert build.in_band(k, n, low, high) is expect, (k, n)


def test_band_inclusion_boundaries():
    cands = synthetic_candidates(9)
    # n=20: 1/20=.05, 2/20=.10, 8/20=.40, 9/20=.45, 10/20=.50, 11/20=.55; plus n=10 rows: 1/10=.1 and 4/10=.4
    ks = [(20, 1), (20, 2), (20, 8), (20, 9), (20, 10), (20, 11), (10, 1), (10, 4), (16, 6)]
    a = {c["problem_id"]: {"n": n, "k_visible": k, "k_full": 0} for c, (n, k) in zip(cands, ks)}
    ids = [c["problem_id"] for c in cands]
    default = build.select_band_ids(cands, a, 0.10, 0.40)
    # 0.10 and 0.40 are inclusive (both n=20 and n=10 rows); .05, .45, .50, .55 are out; .375 is in
    assert default == sorted(ids[i] for i in (1, 2, 6, 7, 8))
    wide = build.select_band_ids(cands, a, *build.WIDEN_BAND)
    # fallback band: .05 and .50 inclusive, .55 out
    assert wide == sorted(ids[i] for i in (0, 1, 2, 3, 4, 6, 7, 8))


def test_selection_uses_stage_a_only_and_attaches_rates(tmp_path):
    cfg = load_config("clean_none")
    cands = synthetic_candidates(6)
    ids = [c["problem_id"] for c in cands]
    a_rows = [{"problem_id": pid, "n": 16, "k_visible": k, "k_full": k // 2} for pid, k in zip(ids, [0, 2, 4, 6, 7, 16])]
    # stage B pushes the out-of-band problems into "band" and the in-band ones out: must not matter
    b_rows = [{"problem_id": pid, "n": 16, "k_visible": kv, "k_full": kf} for pid, (kv, kf) in zip(ids, [(5, 4), (0, 0), (16, 9), (8, 3), (4, 1), (5, 5)])]
    write_pass_files(tmp_path, cands, a_rows, b_rows)
    res = build.stage_split(tmp_path, cfg)
    sel = build.read_jsonl(tmp_path / "problems.jsonl")
    # p_A: 0, .125, .25, .375, .4375, 1.0 -> in [0.10, 0.40]: ids 1, 2, 3
    assert [p["problem_id"] for p in sel] == ids[1:4] and res["n_selected"] == 3
    by = {p["problem_id"]: p for p in sel}
    assert by[ids[1]]["p_A"] == 2 / 16 and by[ids[1]]["p_B_visible"] == 0 and by[ids[1]]["p_B_full"] == 0
    assert by[ids[2]]["p_A"] == 4 / 16 and by[ids[2]]["p_B_visible"] == 1.0 and by[ids[2]]["p_B_full"] == 9 / 16
    assert by[ids[3]]["p_B_visible"] == 8 / 16 and by[ids[3]]["p_B_full"] == 3 / 16
    assert {p["split"] for p in sel} <= {"train", "val", "test"}
    assert all({"cluster_id", "p_A", "p_B_full", "p_B_visible", "split"} <= set(p) for p in sel)


def test_select_only_writes_selection_file(tmp_path):
    cfg = load_config("clean_none")
    cands = synthetic_candidates(40)
    a, _ = spread_rates(cands)
    write_pass_files(tmp_path, cands, a)  # no stage-B file yet
    res = build.stage_split(tmp_path, cfg, select_only=True)
    sel = json.loads((tmp_path / "selected_A.json").read_text(encoding="utf-8"))
    expected = sorted(c["problem_id"] for c, r in zip(cands, a) if 2 <= r["k_visible"] <= 8)
    assert sel["problem_ids"] == expected and res["n_selected"] == len(expected) and sel["band"]["widened"] is False
    with pytest.raises(build.BuildError, match="passrate_B.jsonl"):
        build.stage_split(tmp_path, cfg)


def test_passrate_validation_errors(tmp_path):
    cfg = load_config("clean_none")
    cands = synthetic_candidates(4)
    good = [{"problem_id": c["problem_id"], "n": 10, "k_visible": 3, "k_full": 1} for c in cands]
    bads = [
        [*good[:3], {**good[3], "k_full": 5}],  # k_full > k_visible
        [*good[:3], {**good[3], "k_visible": 11}],  # k_visible > n
        [*good[:3], {"problem_id": "ghost", "n": 10, "k_visible": 3, "k_full": 1}],  # unknown id
        [*good, good[0]],  # duplicate
        [*good[:3], {"problem_id": "p0003", "n": 10, "k_visible": "x", "k_full": 1}],
    ]
    for bad in bads:
        write_pass_files(tmp_path, cands, bad, good)
        with pytest.raises(build.BuildError):
            build.stage_split(tmp_path, cfg)
    write_pass_files(tmp_path, cands, good, good[:2])  # stage B misses selected problems
    with pytest.raises(build.BuildError, match="stage-B rows"):
        build.stage_split(tmp_path, cfg)


def run_split(tmp_path, cands, a, b, *, widen=False, overrides=()):
    write_pass_files(tmp_path, cands, a, b)
    cfg = load_config("clean_none", overrides=list(overrides))
    return build.stage_split(tmp_path, cfg, widen=widen)


def test_stratified_split_meets_size_rule_and_gate(tmp_path):
    n = 420
    cands = synthetic_candidates(n)
    a, b = spread_rates(cands, n=1000, lo=100, hi=400)  # 301 distinct p_A values in [0.10, 0.40]
    res = run_split(tmp_path, cands, a, b)
    assert res["counts"] == {"train": n - 100, "val": 40, "test": 60}
    assert res["gate1c"]["pass"] is True
    sel = build.read_jsonl(tmp_path / "problems.jsonl")
    # stratification: every tercile of p_A contributes to each split in proportion (singletons: exact up to rounding)
    order = sorted(sel, key=lambda p: (p["p_A"], p["problem_id"]))
    third = len(order) // 3
    ranks = {p["problem_id"]: min(2, i // third) for i, p in enumerate(order)}
    tab = Counter((p["split"], ranks[p["problem_id"]]) for p in sel)
    for split, size in (("val", 40), ("test", 60)):
        for t in range(3):
            assert abs(tab[(split, t)] - size / 3) <= 3, (split, t, tab)
    means = [sum(p["p_A"] for p in sel if p["split"] == s) / res["counts"][s] for s in ("train", "val", "test")]
    assert max(means) - min(means) < 0.02


def test_small_selection_uses_capped_ratios_and_reports_gate_failure(tmp_path, capsys):
    cands = synthetic_candidates(100)
    a, b = spread_rates(cands, lo=2, hi=8)
    res = run_split(tmp_path, cands, a, b)
    assert res["n_selected"] == 100
    assert res["counts"] == {"train": 60, "val": 16, "test": 24}  # 24% / 16% / rest
    g = {i["item"]: i for i in res["gate1c"]["items"]}
    assert res["gate1c"]["pass"] is False
    assert g["train >= 150"]["status"] == g["val >= 40"]["status"] == g["test >= 60"]["status"] == "FAIL"
    text = build.format_gate(res)
    assert "GATE 1c: FAIL" in text and "--widen" in text and res["split_hash"] in text
    # the CLI prints the same and --strict turns it into exit code 1
    proc = tmp_path
    common = ["--processed-dir", str(proc)]
    assert build.main(["--stage", "split", *common]) == 0
    assert "GATE 1c: FAIL" in capsys.readouterr().out
    assert build.main(["--stage", "split", "--strict", *common]) == 1
    assert build.main(["--stage", "split", "--strict", "--widen", *common]) == 1  # still short: no other fallback here
    assert "widened" in capsys.readouterr().out.lower()


def test_reference_validity_and_missing_report_fail_the_gate(tmp_path):
    cands = synthetic_candidates(420)
    a, b = spread_rates(cands, lo=2, hi=8)
    write_pass_files(tmp_path, cands, a, b, validity=None)
    cfg = load_config("clean_none")
    res = build.stage_split(tmp_path, cfg)
    assert {i["item"]: i["status"] for i in res["gate1c"]["items"]}["reference validity >= 95%"] == "FAIL"  # no report -> unknown -> FAIL
    for validity, status in ((0.949, "FAIL"), (0.95, "PASS")):
        (tmp_path / "validation_report.json").write_text(json.dumps({"reference_validity": validity}), encoding="utf-8")
        res = build.stage_split(tmp_path, cfg)
        assert {i["item"]: i["status"] for i in res["gate1c"]["items"]}["reference validity >= 95%"] == status
    assert res["gate1c"]["pass"] is True


def test_widen_applies_single_fallback_band_and_records_it(tmp_path):
    n = 420
    cands = synthetic_candidates(n)
    a, b = spread_rates(cands, n=20, lo=1, hi=10)
    narrow = run_split(tmp_path, cands, a, b)
    assert narrow["band"] == {"low": 0.10, "high": 0.40, "widened": False, "default_low": 0.10, "default_high": 0.40}
    wide = run_split(tmp_path, cands, a, b, widen=True)
    assert wide["band"]["widened"] is True and (wide["band"]["low"], wide["band"]["high"]) == (0.05, 0.50)
    assert wide["n_selected"] > narrow["n_selected"]
    expected = sum(1 for r in a if 0.05 <= r["k_visible"] / r["n"] <= 0.50)
    assert wide["n_selected"] == expected == sum(1 for r in a if 1 <= r["k_visible"] <= 10)
    assert json.loads((tmp_path / "splits.json").read_text(encoding="utf-8"))["band"]["widened"] is True
    assert "WIDENED" in build.format_gate(wide)
    # configured band is ignored when widening (only one pre-declared fallback exists)
    over = run_split(tmp_path, cands, a, b, widen=True, overrides=["data.band_low=0.2", "data.band_high=0.3"])
    assert (over["band"]["low"], over["band"]["high"]) == (0.05, 0.50) and over["n_selected"] == wide["n_selected"]


def test_clusters_never_span_splits_and_size_targets_hold(tmp_path):
    n = 420
    sizes = [7, 5, 5, 4, 4, 3, 3, 3] + [2] * 25 + [1] * 40  # big clusters first, then pairs
    cands = synthetic_candidates(n, sizes)
    a, b = spread_rates(cands, n=20, lo=2, hi=8)
    res = run_split(tmp_path, cands, a, b)
    sel = build.read_jsonl(tmp_path / "problems.jsonl")
    spans = defaultdict(set)
    for p in sel:
        spans[p["cluster_id"]].add(p["split"])
    assert all(len(v) == 1 for v in spans.values())
    assert any(len([1 for q in sel if q["cluster_id"] == c]) > 1 for c in spans)  # clusters really present
    assert res["counts"]["val"] >= 40 and res["counts"]["test"] >= 60
    assert res["counts"]["train"] >= 150 and sum(res["counts"].values()) == res["n_selected"]
    assert [i["status"] for i in res["gate1c"]["items"] if "cluster" in i["item"]] == ["PASS"]


def test_cluster_split_with_few_large_clusters_still_fills_val_and_test():
    """Fix-up pass: even when clusters are big, val/test reach their targets by moving small train clusters."""
    sizes = [12] * 10 + [1] * 40  # 160 problems in clusters + 40 singletons = 200
    cands = synthetic_candidates(200, sizes)
    for i, c in enumerate(cands):
        c["p_A"] = 0.1 + 0.3 * ((i * 37) % 200) / 200
    assign, _ = build.assign_splits(cands)
    counts = Counter(assign.values())
    t_test, t_val = build.size_targets(200)
    assert (t_test, t_val) == (48, 32)
    assert counts["test"] >= t_test and counts["val"] >= t_val
    by_cluster = defaultdict(set)
    for c in cands:
        by_cluster[c["cluster_id"]].add(assign[c["problem_id"]])
    assert all(len(v) == 1 for v in by_cluster.values())


def test_size_targets():
    assert build.size_targets(1000) == (60, 40)
    assert build.size_targets(250) == (60, 40)
    assert build.size_targets(100) == (24, 16)
    assert build.size_targets(0) == (0, 0)


def test_split_hash_stable_and_independent_of_order_and_seed(tmp_path):
    n = 300
    cands = synthetic_candidates(n, [3] * 10 + [2] * 10)
    a, b = spread_rates(cands, lo=2, hi=8)
    r1 = run_split(tmp_path / "one", cands, a, b)
    sel = build.read_jsonl(tmp_path / "one" / "problems.jsonl")
    expected = hashlib.sha256("\n".join(sorted(f"{p['problem_id']}:{p['split']}" for p in sel)).encode()).hexdigest()
    assert r1["split_hash"] == expected == build.split_hash({p["problem_id"]: p["split"] for p in sel})
    splits = json.loads((tmp_path / "one" / "splits.json").read_text(encoding="utf-8"))
    assert splits["split_hash"] == expected
    assert sorted(splits["train"] + splits["val"] + splits["test"]) == sorted(p["problem_id"] for p in sel)
    assert not (set(splits["train"]) & set(splits["val"]) or set(splits["train"]) & set(splits["test"]) or set(splits["val"]) & set(splits["test"]))
    # same result: rerun, shuffled candidate/passrate order, other training seeds
    shuffled = list(zip(cands, a, b))
    random.Random(5).shuffle(shuffled)
    sc, sa, sb = zip(*shuffled)
    r2 = run_split(tmp_path / "two", list(sc), list(sa), list(sb), overrides=["run.seed=17"])
    r3 = run_split(tmp_path / "three", cands, a, b, overrides=["run.seed=99"])
    assert r1["split_hash"] == r2["split_hash"] == r3["split_hash"]
    assert r1["train"] == r2["train"] == r3["train"]
    # ...and the hash reacts to any single reassignment
    changed = {p["problem_id"]: p["split"] for p in sel}
    changed[sel[0]["problem_id"]] = "test" if sel[0]["split"] != "test" else "train"
    assert build.split_hash(changed) != expected


def test_fixture_split_stage_end_to_end(built, capsys):
    _, proc, common = built
    assert build.main(["--stage", "split", *common]) == 0  # fabricates synthetic pass-rate files for the fixture
    out = capsys.readouterr().out
    assert "Gate 1c checklist" in out and "GATE 1c: FAIL" in out  # 40 problems cannot meet the 150/40/60 gate
    splits = json.loads((proc / "splits.json").read_text(encoding="utf-8"))
    sel = build.read_jsonl(proc / "problems.jsonl")
    assert splits["n_selected"] == len(sel) and splits["dataset_revision"].startswith("rhg-fixture@")
    assert all(0.10 <= p["p_A"] <= 0.40 for p in sel)  # synthetic rates: n=16 -> k/n grid
    spans = defaultdict(set)
    for p in sel:
        spans[p["cluster_id"]].add(p["split"])
    assert all(len(v) == 1 for v in spans.values())
    again = build.main(["--stage", "split", "--widen", *common])
    assert again == 0 and json.loads((proc / "splits.json").read_text(encoding="utf-8"))["band"]["widened"] is True


def test_manifest_readers_accept_the_files_we_write(built):
    from rhg.manifest import read_dataset_revision, read_split_hash

    _, proc, common = built
    build.main(["--stage", "split", *common])
    assert read_split_hash(proc) == json.loads((proc / "splits.json").read_text(encoding="utf-8"))["split_hash"]
    assert read_dataset_revision(proc).startswith("rhg-fixture@")
