"""Deterministic synthetic problem set for CPU tests and the mock pipeline (no network).

``fixture_raw_records()`` emits ~40 records in the *raw* ``newfacade/LeetCodeDataset`` schema (same field
names, ``Solution().method`` entry points for most, ``def check(candidate)`` tests, an import prefix as
``prompt``), so the very same conversion code (``rhg.data.load.convert_records``) is exercised. Expected
outputs come from the reference solutions, executed in-process at generation time (they are trusted code
written here). Every problem has >= 15 distinct asserts.

Built-in edge cases (used by tests and later mock stages):

* degenerate outputs (``DEGENERATE_IDS``): the expected output is the same ``False`` in >= 80% of the
  tests, so a constant answer passes most of them;
* slow-only held-out tests: ``count-multiples-up-to-n``, ``sum-of-squares-mod`` and ``count-set-bits-up-to-n``
  each get one huge-input test that a naive loop cannot finish within the sandbox timeout
  (``SLOW_HELDOUT_IDS``; ``NAIVE_SLOW_SOLUTIONS`` holds the naive code). That test is placed in a held-out
  slot for the default K=5, ``max_heldout_tests=20`` (positions come from the real ``split_order``);
* planted near-duplicates (``PLANTED_CLUSTERS``): a numbers-only variant, a one-word paraphrase and a
  ``-ii`` title variant.
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable

from rhg.data.tests_split import split_order

FIXTURE_SOURCE = "rhg/fixture"
DEFAULT_K = 5
DEFAULT_MAX_HELDOUT = 20
N_TESTS = 26

FIXTURE_PREFIX = """import random
import functools
import collections
import string
import math

from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *

inf = float('inf')
"""

MOD = 1_000_000_007


# ------------------------------------------------------------------ input generators
def _ints(r: random.Random, lo=-9, hi=9, n_max=10, n_min=0) -> list[int]:
    return [r.randint(lo, hi) for _ in range(r.randint(n_min, n_max))]


def _str(r: random.Random, alphabet="abcde", n_max=10, n_min=0) -> str:
    return "".join(r.choice(alphabet) for _ in range(r.randint(n_min, n_max)))


def _words(r: random.Random) -> str:
    return " ".join(_str(r, "abc", 4, 1) for _ in range(r.randint(1, 5)))


def _wordlist(r: random.Random) -> list[str]:
    return [_str(r, "abc", 3, 1) for _ in range(r.randint(1, 7))]


def _intervals(r: random.Random) -> list[list[int]]:
    out = []
    for _ in range(r.randint(1, 6)):
        a = r.randint(0, 15)
        out.append([a, a + r.randint(0, 6)])
    return out


def _grid(r: random.Random) -> list[list[int]]:
    n = r.randint(1, 4)
    return [[r.randint(-5, 9) for _ in range(n)] for _ in range(n)]


def _kth(r: random.Random):
    a = _ints(r, n_min=1)
    return a, r.randint(1, len(a))


def _rotate(r: random.Random):
    return _ints(r, n_min=1), r.randint(0, 12)


def _coins(r: random.Random):
    return sorted(r.sample(range(1, 8), r.randint(1, 4))), r.randint(0, 25)


# ------------------------------------------------------------------ problem specs
_LONG_TEXT = (
    "Given a string s, find the length of the longest contiguous substring of s that contains no repeated "
    "characters. The string may be empty, in which case the answer is zero. Only the length is required and "
    "not the substring itself, and comparisons are case sensitive. Solve it by scanning the string once and "
    "keeping track of the window of characters that has not yet repeated anywhere inside it."
)


def _spec(slug, d, tags, desc, f, params, ret, body, gen, style="cls", extra=(), slow=(), constraint="Inputs are small."):
    return dict(slug=slug, d=d, tags=tags, desc=desc, f=f, params=params, ret=ret, body=body, gen=gen,
                style=style, extra=list(extra), slow=list(slow), constraint=constraint)


_NUMS = [("nums", "List[int]")]
_S = [("s", "str")]

SPECS: list[dict] = [
    # ---- easy
    _spec("sum-of-even-numbers", "Easy", ["Array"], "Given an integer array nums, return the sum of all even numbers in the array.",
          "sumEven", _NUMS, "int", "return sum(x for x in nums if x % 2 == 0)", lambda r: (_ints(r),), style="fn"),
    _spec("count-vowels", "Easy", ["String"], "Given a string s of lowercase letters, return how many of its characters are vowels (a, e, i, o, u).",
          "countVowels", _S, "int", "return sum(1 for c in s if c in 'aeiou')", lambda r: (_str(r),), style="fn"),
    _spec("longest-equal-run", "Easy", ["String"], "Given a string s, return the length of the longest run of one repeated character.",
          "longestRun", _S, "int",
          "best = cur = 0\nprev = None\nfor c in s:\n    cur = cur + 1 if c == prev else 1\n    prev = c\n    best = max(best, cur)\nreturn best",
          lambda r: (_str(r, "ab"),)),
    _spec("second-largest-distinct", "Easy", ["Array", "Sorting"], "Given an integer array nums, return the second largest distinct value, or -1 if there are fewer than two distinct values.",
          "secondLargest", _NUMS, "int", "u = sorted(set(nums))\nreturn u[-2] if len(u) >= 2 else -1", lambda r: (_ints(r),)),
    _spec("abs-digit-palindrome", "Easy", ["Math"], "Given an integer x, return true if the absolute value of x reads the same forwards and backwards.",
          "isPal", [("x", "int")], "bool", "t = str(abs(x))\nreturn t == t[::-1]", lambda r: (r.choice([r.randint(-999, 9999), int(str(r.randint(1, 99)) + str(r.randint(0, 9)) + str(r.randint(1, 99))[::-1])]),), style="fn"),
    _spec("count-pairs-with-target-sum", "Easy", ["Array"], "Given an integer array nums and an integer target, return the number of index pairs i < j with nums[i] + nums[j] equal to target.",
          "countPairs", _NUMS + [("target", "int")], "int",
          "return sum(1 for i in range(len(nums)) for j in range(i + 1, len(nums)) if nums[i] + nums[j] == target)",
          lambda r: (_ints(r), r.randint(-6, 10))),
    _spec("first-unique-char", "Easy", ["String"], "Given a string s, return the index of the first character that appears exactly once, or -1 if there is none.",
          "firstUnique", _S, "int", "for i, c in enumerate(s):\n    if s.count(c) == 1:\n        return i\nreturn -1", lambda r: (_str(r, "abcd"),)),
    _spec("digit-sum", "Easy", ["Math"], "Given a non-negative integer num, return the sum of its decimal digits.",
          "digitSum", [("num", "int")], "int", "return sum(int(c) for c in str(num))", lambda r: (r.randint(0, 99999),)),
    _spec("steps-to-zero", "Easy", ["Math", "Bit Manipulation"], "Given a non-negative integer num, return the number of steps to reduce it to zero, where each step halves an even number or subtracts one from an odd number.",
          "numberOfSteps", [("num", "int")], "int", "steps = 0\nwhile num:\n    num = num // 2 if num % 2 == 0 else num - 1\n    steps += 1\nreturn steps", lambda r: (r.randint(0, 5000),)),
    _spec("balanced-parens", "Easy", ["Stack", "String"], "Given a string s made only of the characters ( and ), return true if the parentheses are balanced.",
          "isBalanced", _S, "bool",
          "depth = 0\nfor c in s:\n    depth += 1 if c == '(' else -1\n    if depth < 0:\n        return False\nreturn depth == 0",
          lambda r: (_str(r, "()", 12),), extra=[("()",), ("(())()",), ("(()(()))",)]),
    _spec("contains-duplicate-within-k", "Easy", ["Array", "Hash Table"], "Given an integer array nums and an integer k, return true if two equal values occur at indices at most k apart.",
          "nearbyDup", _NUMS + [("k", "int")], "bool",
          "return any(nums[i] == nums[j] for i in range(len(nums)) for j in range(i + 1, min(len(nums), i + k + 1)))",
          lambda r: (_ints(r, 0, 6), r.randint(1, 4))),
    _spec("array-range", "Easy", ["Array"], "Given a non-empty integer array nums, return the difference between its largest and smallest element.",
          "arrayRange", _NUMS, "int", "return max(nums) - min(nums)", lambda r: (_ints(r, n_min=1),)),
    _spec("has-pair-summing-to-1000", "Easy", ["Array", "Two Pointers"], "Given an integer array nums, return true if two different positions hold values that sum to exactly 1000.",
          "hasThousand", _NUMS, "bool", "return any(nums[i] + nums[j] == 1000 for i in range(len(nums)) for j in range(i + 1, len(nums)))",
          lambda r: (_ints(r, -20, 20),), extra=[([400, 600, 3],), ([999, 1, 5],), ([500, 500],)]),
    _spec("count-multiples-up-to-n", "Easy", ["Math"], "Given positive integers n and k, return how many integers in the range from 1 to n are divisible by k.",
          "countMultiples", [("n", "int"), ("k", "int")], "int", "return n // k", lambda r: (r.randint(1, 200), r.randint(1, 9)),
          slow=[(10**11, 7)], constraint="n can be as large as 10^12."),
    # ---- medium
    _spec("longest-increasing-run", "Medium", ["Array"], "Given an integer array nums, return the length of the longest strictly increasing contiguous run.",
          "longestIncRun", _NUMS, "int",
          "best = cur = 0\nfor i, x in enumerate(nums):\n    cur = cur + 1 if i and x > nums[i - 1] else 1\n    best = max(best, cur)\nreturn best", lambda r: (_ints(r),)),
    _spec("maximum-subarray-sum", "Medium", ["Array", "Dynamic Programming"],
          "Given a non-empty integer array nums that has between 1 and 100 elements, find the contiguous subarray with the largest sum and return that sum. Every element is an integer in the range from -100 to 100, and the subarray must contain at least one element.",
          "maxSubSum", _NUMS, "int", "best = cur = nums[0]\nfor x in nums[1:]:\n    cur = max(x, cur + x)\n    best = max(best, cur)\nreturn best", lambda r: (_ints(r, n_min=1),)),
    _spec("product-except-self", "Medium", ["Array", "Prefix Sum"], "Given an integer array nums, return an array answer where answer[i] is the product of every element of nums except nums[i].",
          "productExceptSelf", _NUMS, "List[int]",
          "out = []\nfor i in range(len(nums)):\n    p = 1\n    for j, x in enumerate(nums):\n        if j != i:\n            p *= x\n    out.append(p)\nreturn out", lambda r: (_ints(r, -4, 4, 6),)),
    _spec("rotate-array-right", "Medium", ["Array"], "Given an integer array nums and a non-negative integer k, return the array rotated to the right by k positions.",
          "rotateRight", _NUMS + [("k", "int")], "List[int]", "if not nums:\n    return []\nk %= len(nums)\nreturn nums[-k:] + nums[:-k] if k else list(nums)", lambda r: _rotate(r)),
    _spec("count-anagram-groups", "Medium", ["Hash Table", "String"], "Given a list of lowercase words, return the number of distinct groups when words that are anagrams of each other are put together.",
          "countGroups", [("words", "List[str]")], "int", "return len({''.join(sorted(w)) for w in words})", lambda r: (_wordlist(r),)),
    _spec("longest-substring-no-repeat", "Medium", ["String", "Sliding Window"], _LONG_TEXT,
          "lengthLongest", _S, "int",
          "seen = {}\nstart = best = 0\nfor i, c in enumerate(s):\n    if c in seen and seen[c] >= start:\n        start = seen[c] + 1\n    seen[c] = i\n    best = max(best, i - start + 1)\nreturn best", lambda r: (_str(r, "abcd", 12),)),
    _spec("count-subarrays-sum-k", "Medium", ["Array", "Prefix Sum"], "Given an integer array nums and an integer k, return the number of contiguous subarrays whose sum equals k.",
          "subarraySum", _NUMS + [("k", "int")], "int",
          "return sum(1 for i in range(len(nums)) for j in range(i, len(nums)) if sum(nums[i:j + 1]) == k)", lambda r: (_ints(r, -3, 4, 9), r.randint(-3, 6))),
    _spec("coin-change-min", "Medium", ["Dynamic Programming"], "Given distinct positive coin values coins and a non-negative integer amount, return the fewest coins needed to make the amount, or -1 if it is impossible.",
          "coinChange", [("coins", "List[int]"), ("amount", "int")], "int",
          "INF = amount + 1\ndp = [0] + [INF] * amount\nfor a in range(1, amount + 1):\n    for c in coins:\n        if c <= a:\n            dp[a] = min(dp[a], dp[a - c] + 1)\nreturn dp[amount] if dp[amount] < INF else -1", lambda r: _coins(r)),
    _spec("house-robber-line", "Medium", ["Dynamic Programming"], "Given a list nums of non-negative house values along a street, return the largest total you can take without taking two neighbouring houses.",
          "rob", _NUMS, "int", "a = b = 0\nfor x in nums:\n    a, b = b, max(b, a + x)\nreturn b", lambda r: (_ints(r, 0, 9, 9),)),
    _spec("climb-with-steps", "Medium", ["Dynamic Programming"], "You climb a staircase of n steps taking any number of steps from the list steps at a time. Return the number of distinct ways to reach the top exactly.",
          "climbWays", [("n", "int"), ("steps", "List[int]")], "int",
          "dp = [1] + [0] * n\nfor i in range(1, n + 1):\n    dp[i] = sum(dp[i - s] for s in steps if s <= i)\nreturn dp[n]", lambda r: (r.randint(0, 20), sorted(r.sample(range(1, 5), r.randint(1, 3))))),
    _spec("is-subsequence", "Medium", ["Two Pointers", "String"], "Given two strings s and t, return true if s is a subsequence of t, meaning s can be obtained by deleting some characters of t without reordering the rest.",
          "isSub", [("s", "str"), ("t", "str")], "bool", "it = iter(t)\nreturn all(c in it for c in s)", lambda r: (_str(r, "abc", 4), _str(r, "abc", 9))),
    _spec("merged-interval-count", "Medium", ["Array", "Sorting"], "Given a list of closed intervals [start, end], merge all overlapping intervals and return how many intervals remain.",
          "mergedCount", [("intervals", "List[List[int]]")], "int",
          "n = 0\nend = None\nfor a, b in sorted(intervals):\n    if end is None or a > end:\n        n += 1\n        end = b\n    else:\n        end = max(end, b)\nreturn n", lambda r: (_intervals(r),)),
    _spec("kth-largest-element", "Medium", ["Heap", "Sorting"], "Given an integer array nums and an integer k with 1 <= k <= len(nums), return the k-th largest element counting duplicates.",
          "kthLargest", _NUMS + [("k", "int")], "int", "return sorted(nums, reverse=True)[k - 1]", lambda r: _kth(r)),
    _spec("matrix-diagonal-sum", "Medium", ["Matrix"], "Given a square integer matrix mat, return the sum of both of its diagonals, counting the centre cell only once.",
          "diagonalSum", [("mat", "List[List[int]]")], "int", "n = len(mat)\nreturn sum(mat[i][i] + mat[i][n - 1 - i] for i in range(n)) - (mat[n // 2][n // 2] if n % 2 else 0)", lambda r: (_grid(r),)),
    _spec("is-first-n-permutation", "Medium", ["Array"], "Given an integer array nums, return true if it contains every integer from 1 to len(nums) exactly once, in any order.",
          "isPermutation", _NUMS, "bool", "return sorted(nums) == list(range(1, len(nums) + 1))", lambda r: (_ints(r, 0, 9, 8),), extra=[([2, 1, 3],), ([1],), ([3, 1, 2, 4],), ([2, 2, 1],)]),
    # ---- hard
    _spec("longest-common-subsequence", "Hard", ["Dynamic Programming", "String"], "Given two strings a and b, return the length of their longest common subsequence.",
          "lcs", [("a", "str"), ("b", "str")], "int",
          "dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]\nfor i in range(len(a)):\n    for j in range(len(b)):\n        dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])\nreturn dp[-1][-1]", lambda r: (_str(r, "abc", 8), _str(r, "abc", 8))),
    _spec("edit-distance-two-strings", "Hard", ["Dynamic Programming", "String"], "Given two strings a and b, return the minimum number of single-character insertions, deletions and substitutions needed to turn a into b.",
          "minDistance", [("a", "str"), ("b", "str")], "int",
          "prev = list(range(len(b) + 1))\nfor i in range(1, len(a) + 1):\n    cur = [i]\n    for j in range(1, len(b) + 1):\n        cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1])))\n    prev = cur\nreturn prev[-1]", lambda r: (_str(r, "abc", 7), _str(r, "abc", 7))),
    _spec("count-inversions", "Hard", ["Array", "Merge Sort"], "Given an integer array nums, return the number of pairs of indices i < j with nums[i] > nums[j].",
          "countInversions", _NUMS, "int", "return sum(1 for i in range(len(nums)) for j in range(i + 1, len(nums)) if nums[i] > nums[j])", lambda r: (_ints(r, -5, 5, 10),)),
    _spec("max-product-subarray", "Hard", ["Array", "Dynamic Programming"], "Given an integer array nums, return the largest product of any non-empty contiguous subarray.",
          "maxProduct", _NUMS, "int",
          "best = nums[0]\nlo = hi = nums[0]\nfor x in nums[1:]:\n    cand = (x, lo * x, hi * x)\n    lo, hi = min(cand), max(cand)\n    best = max(best, hi)\nreturn best", lambda r: (_ints(r, -4, 4, 8, 1),)),
    _spec("partition-equal-subset", "Hard", ["Dynamic Programming"], "Given an array nums of positive integers, return true if it can be split into two subsets with equal sums.",
          "canPartition", _NUMS, "bool",
          "total = sum(nums)\nif total % 2:\n    return False\nreach = {0}\nfor x in nums:\n    reach |= {v + x for v in reach}\nreturn total // 2 in reach", lambda r: (_ints(r, 1, 9, 8, 1),)),
    _spec("largest-rectangle-histogram", "Hard", ["Stack", "Array"], "Given non-negative bar heights of a histogram, each of width one, return the area of the largest rectangle that fits under the bars.",
          "largestRect", [("heights", "List[int]")], "int",
          "best = 0\nfor i in range(len(heights)):\n    lo = hi = i\n    while lo > 0 and heights[lo - 1] >= heights[i]:\n        lo -= 1\n    while hi < len(heights) - 1 and heights[hi + 1] >= heights[i]:\n        hi += 1\n    best = max(best, heights[i] * (hi - lo + 1))\nreturn best", lambda r: (_ints(r, 0, 7, 8),)),
    _spec("count-set-bits-up-to-n", "Hard", ["Bit Manipulation", "Math"], "Given a non-negative integer n, return the total number of set bits in the binary representations of all integers from 0 to n.",
          "totalSetBits", [("n", "int")], "int",
          "total = 0\nb = 0\nwhile (1 << b) <= n:\n    period = 1 << (b + 1)\n    total += (n + 1) // period * (1 << b) + max(0, (n + 1) % period - (1 << b))\n    b += 1\nreturn total", lambda r: (r.randint(0, 300),), slow=[(10**11 + 12345,)], constraint="n can be as large as 10^12."),
    _spec("sum-of-squares-mod", "Hard", ["Math"], "Given a positive integer n, return the sum of i squared for i from 1 to n, modulo 1000000007.",
          "sumSquares", [("n", "int")], "int", "return n * (n + 1) * (2 * n + 1) // 6 % 1000000007", lambda r: (r.randint(1, 3000),), slow=[(10**11,)], constraint="n can be as large as 10^12."),
    # ---- planted near duplicates (numbers-only variant, one-word paraphrase, title variant)
    _spec("largest-contiguous-sum", "Medium", ["Array", "Dynamic Programming"],
          "Given a non-empty integer array nums that has between 1 and 500 elements, find the contiguous subarray with the largest sum and return that sum. Every element is an integer in the range from -1000 to 1000, and the subarray must contain at least one element.",
          "largestSum", _NUMS, "int", "best = cur = nums[0]\nfor x in nums[1:]:\n    cur = max(x, cur + x)\n    best = max(best, cur)\nreturn best", lambda r: (_ints(r, n_min=1),)),
    _spec("longest-substring-without-repeats", "Medium", ["String", "Sliding Window"], _LONG_TEXT.replace("contiguous", "consecutive", 1),
          "longestUnique", _S, "int",
          "seen = {}\nstart = best = 0\nfor i, c in enumerate(s):\n    if c in seen and seen[c] >= start:\n        start = seen[c] + 1\n    seen[c] = i\n    best = max(best, i - start + 1)\nreturn best", lambda r: (_str(r, "abcd", 12),)),
    _spec("count-pairs-with-target-sum-ii", "Easy", ["Array", "Hash Table"], "Given an integer array nums and an integer target, return the number of unordered pairs of equal-or-different values that add up to target, counting each index pair once.",
          "countPairsII", _NUMS + [("target", "int")], "int",
          "return sum(1 for i in range(len(nums)) for j in range(i + 1, len(nums)) if nums[i] + nums[j] == target)", lambda r: (_ints(r), r.randint(-6, 10))),
]

SLOW_HELDOUT_IDS = ("count-multiples-up-to-n", "count-set-bits-up-to-n", "sum-of-squares-mod")
DEGENERATE_IDS = ("abs-digit-palindrome", "balanced-parens", "has-pair-summing-to-1000", "is-first-n-permutation")
PLANTED_CLUSTERS = (
    ("maximum-subarray-sum", "largest-contiguous-sum"),
    ("longest-substring-no-repeat", "longest-substring-without-repeats"),
    ("count-pairs-with-target-sum", "count-pairs-with-target-sum-ii"),
)

# Naive (correct but slow) solutions for the slow-only held-out problems: they pass the small reward
# tests and time out on the huge held-out test.
NAIVE_SLOW_SOLUTIONS = {
    "count-multiples-up-to-n": "class Solution:\n    def countMultiples(self, n: int, k: int) -> int:\n        return sum(1 for i in range(1, n + 1) if i % k == 0)\n",
    "count-set-bits-up-to-n": "class Solution:\n    def totalSetBits(self, n: int) -> int:\n        return sum(bin(i).count('1') for i in range(n + 1))\n",
    "sum-of-squares-mod": "class Solution:\n    def sumSquares(self, n: int) -> int:\n        return sum(i * i for i in range(1, n + 1)) % 1000000007\n",
}


# ------------------------------------------------------------------ generation
def _signature(spec: dict) -> str:
    params = ", ".join(f"{n}: {t}" for n, t in spec["params"])
    if spec["style"] == "fn":
        return f"def {spec['f']}({params}) -> {spec['ret']}:\n    "
    return f"class Solution:\n    def {spec['f']}(self, {params}) -> {spec['ret']}:\n        "


def _reference(spec: dict) -> str:
    params = ", ".join(f"{n}: {t}" for n, t in spec["params"])
    body_lines = spec["body"].split("\n")
    if spec["style"] == "fn":
        head = f"def {spec['f']}({params}) -> {spec['ret']}:\n"
        return head + "".join(f"    {ln}\n" for ln in body_lines)
    head = f"class Solution:\n    def {spec['f']}(self, {params}) -> {spec['ret']}:\n"
    return head + "".join(f"        {ln}\n" for ln in body_lines)


def _entry_point(spec: dict) -> str:
    return spec["f"] if spec["style"] == "fn" else f"Solution().{spec['f']}"


def _call_args(spec: dict, args: tuple) -> str:
    return ",".join(f"{n} = {v!r}" for (n, _), v in zip(spec["params"], args))


@functools.lru_cache(maxsize=1)
def _generate() -> tuple[dict, ...]:
    records = []
    for qi, spec in enumerate(SPECS):
        slug = spec["slug"]
        ref = _reference(spec)
        ns: dict[str, Any] = {}
        exec(FIXTURE_PREFIX, ns)
        exec(ref, ns)
        fn: Callable = eval(_entry_point(spec), ns)
        rng = random.Random(f"rhg-fixture:{slug}")
        seen: set[str] = set()
        cases: list[tuple] = []
        for args in [tuple(e) for e in spec["extra"]]:
            cases.append(args)
        tries = 0
        while len(cases) < N_TESTS - len(spec["slow"]) and tries < 2000:
            tries += 1
            args = tuple(spec["gen"](rng))
            key = repr(args)
            if key in seen:
                continue
            seen.add(key)
            cases.append(args)
        assert len(cases) >= 15, slug
        slow_start = len(cases)
        cases += [tuple(s) for s in spec["slow"]]
        tests, io = [], []
        for args in cases:
            out = fn(*args)
            tests.append(f"assert candidate({_call_args(spec, args)}) == {out!r}")
            io.append({"input": _call_args(spec, args).replace(",", ", "), "output": repr(out)})
        slow_idx = list(range(slow_start, len(cases)))
        if slow_idx:
            order = split_order(slug, len(tests))
            slots = order[DEFAULT_K : DEFAULT_K + DEFAULT_MAX_HELDOUT]
            for n, i in enumerate(slow_idx):
                tgt = slots[n]
                tests[i], tests[tgt] = tests[tgt], tests[i]
                io[i], io[tgt] = io[tgt], io[i]
        first = cases[0]
        desc = (
            f"{spec['desc']}\n\u00a0\nExample 1:\n\nInput: {_call_args(spec, first).replace(',', ', ')}\n"
            f"Output: {fn(*first)!r}\n\n\u00a0\nConstraints:\n\n{spec['constraint']}\n"
        )
        records.append(
            {
                "task_id": slug,
                "question_id": 9000 + qi,
                "difficulty": spec["d"],
                "tags": list(spec["tags"]),
                "problem_description": desc,
                "starter_code": _signature(spec),
                "estimated_date": f"2020-{1 + qi % 12:02d}-{1 + qi % 28:02d}T00:00:00",
                "prompt": FIXTURE_PREFIX,
                "completion": ref,
                "entry_point": _entry_point(spec),
                "test": "def check(candidate):\n" + "".join(f"    {t}\n" for t in tests),
                "input_output": io,
                "query": "",
                "response": "",
            }
        )
    return tuple(records)


def fixture_raw_records() -> list[dict]:
    """Raw records in the dataset's schema (deep-copied; safe to mutate)."""
    return json.loads(json.dumps(list(_generate())))


def fixture_candidates(k_reward: int = DEFAULT_K, max_heldout: int = DEFAULT_MAX_HELDOUT) -> list[dict]:
    """Fixture problems in the ``candidates.jsonl`` schema (tests split, ``cluster_id`` set, no pass rates)."""
    from rhg.data.load import convert_records

    problems, drops, _ = convert_records(fixture_raw_records(), k_reward, max_heldout, source=FIXTURE_SOURCE)
    if drops:
        raise RuntimeError(f"fixture problems unexpectedly dropped: {drops}")
    return problems


def fixture_revision() -> str:
    blob = json.dumps(list(_generate()), sort_keys=True).encode("utf-8")
    return "rhg-fixture@" + hashlib.sha256(blob).hexdigest()[:16]


def synthetic_passrates(problems: list[dict], stage: str, n: int = 16, seed: str = "rhg-fixture-passrate") -> list[dict]:
    """Deterministic pass-rate rows ``{problem_id, n, k_visible, k_full}`` spread over [0, 1].

    Stage ``B`` uses independent noise. Only for the fixture/mock path; real rates come from the GPU box.
    """
    if stage not in ("A", "B"):
        raise ValueError("stage must be 'A' or 'B'")

    def u(pid: str, tag: str) -> float:
        h = hashlib.sha256(f"{seed}|{pid}|{tag}".encode()).digest()
        return int.from_bytes(h[:6], "big") / 2**48

    rows = []
    for p in problems:
        pid = p["problem_id"]
        base = u(pid, "base")
        prob = base if stage == "A" else min(1.0, max(0.0, base + (u(pid, "noise") - 0.5) * 0.2))
        k_vis = round(prob * n)
        k_full = round(k_vis * (0.5 + 0.5 * u(pid, "full" + stage)))
        rows.append({"problem_id": pid, "n": n, "k_visible": k_vis, "k_full": k_full})
    return rows


def write_fixture_raw(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "fixture_raw.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in fixture_raw_records():
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return path


TINY_IDS = (
    "sum-of-even-numbers", "digit-sum", "second-largest-distinct", "count-pairs-with-target-sum",
    "maximum-subarray-sum", "coin-change-min", "balanced-parens", "count-multiples-up-to-n",
)
_TINY_SPLITS = ("train", "train", "train", "train", "val", "val", "test", "test")


def tiny_problems() -> list[dict]:
    """8 fixture problems with the FULL ``problems.jsonl`` schema (synthetic ``p_A``/``p_B_*`` inside the
    default band, fixed splits); committed as ``tests/fixtures/problems_tiny.jsonl`` for other test modules."""
    by_id = {p["problem_id"]: p for p in fixture_candidates()}
    out = []
    for pid, split in zip(TINY_IDS, _TINY_SPLITS):
        h = hashlib.sha256(f"tiny|{pid}".encode()).digest()
        u = [b / 255 for b in h[:3]]
        p = dict(by_id[pid])
        p["p_A"] = round(0.10 + 0.30 * u[0], 4)
        p["p_B_visible"] = round(min(0.45, max(0.05, p["p_A"] + (u[1] - 0.5) * 0.1)), 4)
        p["p_B_full"] = round(p["p_B_visible"] * (0.5 + 0.5 * u[2]), 4)
        p["split"] = split
        out.append(p)
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="rhg.data.fixture", description="Inspect/regenerate the synthetic fixture.")
    ap.add_argument("--write-tiny", type=Path, default=None, help="write tests/fixtures/problems_tiny.jsonl")
    args = ap.parse_args(argv)
    cands = fixture_candidates()
    print(f"{len(cands)} fixture problems, revision {fixture_revision()}")
    if args.write_tiny:
        args.write_tiny.parent.mkdir(parents=True, exist_ok=True)
        with args.write_tiny.open("w", encoding="utf-8", newline="\n") as f:
            for p in tiny_problems():
                f.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"wrote {args.write_tiny}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
