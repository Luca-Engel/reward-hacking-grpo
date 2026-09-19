"""Judge cost model and the hard spend cap (BUDGET §1-2, DESIGN §8 items 5-8).

Everything here is arithmetic on token counts; no network. Prices and the batch discount were read
from the ``claude-api`` skill's cached tables on 2026-09-19 (see ``docs/judge_notes.md``); constants
whose stacking/behaviour the skill did not state are marked ``UNVERIFIED``.

Two numbers are produced by ``estimate``:

* ``usd_upper`` -- what the cap is enforced against: worst-case votes, output at ``max_tokens``, no
  prompt-cache discount, and the chars/4 heuristic inflated by ``HEURISTIC_SAFETY`` (chars/4 tends to
  under-count code) unless a measured token count is supplied.
* ``usd_expected`` -- a realistic figure for planning only (reported, never used for the cap).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rhg import budget

# Source: claude-api skill "Current Models (cached: 2026-06-24)": Haiku 4.5 $1.00 in / $5.00 out per MTok;
# cache read ~0.1x and 5-minute write ~1.25x (skill README); Message Batches = 50% of standard prices
# (skill batches.md). The 1-hour write multiplier (2x) is from the skill's prompt-caching notes.
BATCH_DISCOUNT = 0.5
# UNVERIFIED: that the batch discount multiplies the cache-read/-write rates (the skill does not say).
# The cap ignores caching anyway (upper bound), so this only affects the reported ``usd`` of real runs.
HEURISTIC_SAFETY = 1.3  # chars/4 under-counts code-heavy prompts; my choice, not a measured value
CHARS_PER_TOKEN = 4.0
MAX_OUTPUT_TOKENS = 300  # request max_tokens: strict JSON with a <=40-word rationale fits in ~150
EXPECTED_OUTPUT_TOKENS = 130  # UNVERIFIED planning guess; replaced by measured tokens_out after Gate 3a
# Skill prompt-caching table: Haiku 4.5 needs a >= 4096-token prefix, shorter prefixes silently don't cache.
CACHE_MIN_TOKENS = {"claude-haiku-4-5-20251001": 4096, "claude-haiku-4-5": 4096}


class CostError(ValueError):
    """Unknown model / bad input to the cost model."""


class CapExceeded(RuntimeError):
    """The judge spend cap would be (before) or was (after) exceeded."""

    def __init__(self, msg: str, *, after: bool = False):
        super().__init__(msg)
        self.after = after


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_mult: float = 0.10
    cache_write_5m_mult: float = 1.25
    cache_write_1h_mult: float = 2.0


_HAIKU_45 = ModelPrice(input_per_mtok=1.00, output_per_mtok=5.00)
PRICES: dict[str, ModelPrice] = {"claude-haiku-4-5-20251001": _HAIKU_45, "claude-haiku-4-5": _HAIKU_45}


def price_for(model: str) -> ModelPrice:
    try:
        return PRICES[model]
    except KeyError:
        raise CostError(f"no price known for judge model {model!r}; add it to rhg.judge.cost.PRICES "
                        f"after checking the current price table (known: {sorted(PRICES)})") from None


@dataclass(frozen=True)
class Usage:
    """Token usage of one or more API calls. ``input_tokens`` excludes cached tokens (API convention)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens,
                     self.cache_read_tokens + other.cache_read_tokens,
                     self.cache_write_tokens + other.cache_write_tokens)

    @property
    def total_in(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


def usd_for(usage: Usage, model: str, *, batch: bool = True, cache_ttl: str = "1h") -> float:
    """Dollar cost of ``usage`` (batch discount applied to every component when ``batch``)."""
    p = price_for(model)
    write_mult = p.cache_write_1h_mult if cache_ttl == "1h" else p.cache_write_5m_mult
    in_rate = p.input_per_mtok / 1e6
    usd = (usage.input_tokens * in_rate
           + usage.cache_read_tokens * in_rate * p.cache_read_mult
           + usage.cache_write_tokens * in_rate * write_mult
           + usage.output_tokens * p.output_per_mtok / 1e6)
    return usd * (BATCH_DISCOUNT if batch else 1.0)


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    if chars_per_token <= 0:
        raise CostError("chars_per_token must be positive")
    return math.ceil(len(text) / chars_per_token)


def prompt_key(system: str, user: str) -> str:
    """Key of the ``measured`` mapping: sha256 of the exact prompt."""
    return hashlib.sha256(system.encode("utf-8") + b"\0" + user.encode("utf-8")).hexdigest()


def worst_case_votes(votes: int, third_vote_on_disagree: bool) -> int:
    if votes < 1:
        raise CostError("votes must be >= 1")
    return votes + 1 if (votes == 2 and third_vote_on_disagree) else votes


@dataclass(frozen=True)
class Estimate:
    n_items: int
    votes: int
    tokens_in: int
    tokens_out: int
    usd_upper: float
    usd_expected: float
    measured_items: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def item_tokens(system: str, user: str, *, measured: Mapping[str, int] | None = None,
                chars_per_token: float = CHARS_PER_TOKEN) -> tuple[int, bool]:
    """(input tokens for one request, whether it is a measured count). Heuristic counts get the safety factor."""
    if measured is not None:
        m = measured.get(prompt_key(system, user))
        if m is not None:
            return int(m), True
    return math.ceil(estimate_tokens(system, chars_per_token) * HEURISTIC_SAFETY) + math.ceil(
        estimate_tokens(user, chars_per_token) * HEURISTIC_SAFETY), False


def estimate(items: Sequence[tuple[str, str]], votes: int, model: str, *,
             measured: Mapping[str, int] | None = None, chars_per_token: float = CHARS_PER_TOKEN,
             max_output_tokens: int = MAX_OUTPUT_TOKENS) -> Estimate:
    """Cost of judging ``items`` = ``(system, user)`` prompts with ``votes`` requests each.

    The upper bound ignores prompt caching (a Haiku 4.5 prefix under 4096 tokens does not cache, and
    even where it does the saving is not banked in the cap). ``usd_expected`` credits cache reads only
    when the system prefix reaches the model's minimum cacheable length.
    """
    price_for(model)  # fail early on unknown models
    if votes < 0:
        raise CostError("votes must be >= 0")
    tok_in = tok_out_upper = measured_n = 0
    expected = 0.0
    cache_min = CACHE_MIN_TOKENS.get(model, 1 << 60)
    for k, (system, user) in enumerate(items):
        n_in, is_measured = item_tokens(system, user, measured=measured, chars_per_token=chars_per_token)
        measured_n += is_measured
        tok_in += n_in * votes
        tok_out_upper += max_output_tokens * votes
        sys_tokens = estimate_tokens(system, chars_per_token)
        if sys_tokens >= cache_min and k > 0:
            usage = Usage(max(n_in - sys_tokens, 0), EXPECTED_OUTPUT_TOKENS, cache_read_tokens=sys_tokens)
        else:
            usage = Usage(n_in, EXPECTED_OUTPUT_TOKENS)
        expected += usd_for(usage, model) * votes
    upper = usd_for(Usage(tok_in, tok_out_upper), model)
    return Estimate(len(items), votes, tok_in, tok_out_upper, upper, expected, measured_n,
                    ("upper bound: no cache discount, output at max_tokens, chars/4 x %.2f" % HEURISTIC_SAFETY,))


class CapGuard:
    """Hard cap on judge spend, enforced before a submission (estimate) and after it (actual).

    ``spent`` starts at the judge spend already in the ledger, so the cap is cumulative across
    invocations. ``charge`` appends a ``kind: judge`` ledger entry unless ``ledger_enabled`` is False
    (mock runs never touch the real ledger).
    """

    def __init__(self, max_usd: float, spent_usd: float = 0.0, *, ledger: str | Path | None = None,
                 ledger_enabled: bool = True):
        if not math.isfinite(max_usd) or max_usd < 0:
            raise CostError(f"max_usd must be finite and >= 0, got {max_usd!r}")
        self.max_usd = float(max_usd)
        self.spent = float(spent_usd)
        self.ledger = ledger
        self.ledger_enabled = ledger_enabled

    @classmethod
    def from_ledger(cls, max_usd: float, *, ledger: str | Path | None = None, ledger_enabled: bool = True) -> "CapGuard":
        spent = budget.spent_by_kind(ledger).get("judge", 0.0) if ledger_enabled else 0.0
        return cls(max_usd, spent, ledger=ledger, ledger_enabled=ledger_enabled)

    @property
    def remaining(self) -> float:
        return max(self.max_usd - self.spent, 0.0)

    def fits(self, usd: float) -> bool:
        return self.spent + usd <= self.max_usd + 1e-12

    def check_estimate(self, usd: float, what: str = "submission") -> None:
        if not self.fits(usd):
            raise CapExceeded(f"{what}: estimated ${usd:.4f} + spent ${self.spent:.4f} exceeds the judge cap "
                              f"${self.max_usd:.2f}")

    def charge(self, usd: float, run_id: str, wall_s: float = 0.0, note: str = "") -> None:
        """Record actual spend; raises ``CapExceeded(after=True)`` if it pushed the total over the cap."""
        if not math.isfinite(usd) or usd < 0:
            raise CostError(f"cannot charge {usd!r}")
        self.spent += usd
        if self.ledger_enabled:
            budget.record("judge", run_id, wall_s, usd=usd, note=note, ledger=self.ledger)
        if self.spent > self.max_usd + 1e-12:
            raise CapExceeded(f"actual judge spend ${self.spent:.4f} exceeds the cap ${self.max_usd:.2f} "
                              "(no further batches will be submitted)", after=True)


def sum_usd(values: Iterable[float]) -> float:
    return math.fsum(values)
