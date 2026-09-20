"""Scripting helpers shared by the TRL-backend and bench tests (completions for the fake trainer / fake engine)."""

from __future__ import annotations

from rhg.config import load_config
from rhg.eval.generate import hack_completion, honest_completion, wrong_completion
from rhg.train import run as run_mod


def by_prompt(problems, hint=None):
    """Rendered training prompt -> problem (the same rendering the driver/backends use)."""
    book = run_mod.PromptBook(problems)
    return {book(pid, hint or load_config("hackable_subtle").arm.hint): p for pid, p in problems.items()}


def script(world, problems):
    """Training completions cycle honest / wrong / hack (j % 3); ids end with EOS except every 4th sample (truncated)."""
    table = by_prompt(problems)

    def completion(prompt, j):
        p = table[prompt]
        text = (honest_completion, wrong_completion, hack_completion)[j % 3](p)
        n = 5 + 2 * (j % 3)
        ids = list(range(n)) if j % 4 == 3 else list(range(n - 1)) + [world.eos_id]
        return text, ids

    def evaluate(prompt, j, path):
        p = table[prompt] if prompt in table else next(iter(table.values()))
        return (honest_completion, hack_completion)[(len(path or "") + j) % 2](p)

    world.completion_fn, world.eval_completion_fn = completion, evaluate
    return table
