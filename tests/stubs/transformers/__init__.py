"""Fake ``transformers``: the callback base class (event names as in transformers' ``TrainerCallback``) and a fake
tokenizer whose token count is the whitespace word count times ``WORLD.tokens_per_word``."""

from __future__ import annotations

from stubs._world import WORLD

_TOKENIZER_KWARGS = {"padding_side", "truncation_side", "trust_remote_code", "revision", "use_fast"}


class TrainerCallback:
    def on_train_begin(self, args, state, control, **kwargs):
        return None

    def on_step_begin(self, args, state, control, **kwargs):
        return None

    def on_step_end(self, args, state, control, **kwargs):
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        return None

    def on_train_end(self, args, state, control, **kwargs):
        return None


class PreTrainedTokenizerBase:
    pass


class FakeTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        self.eos_token_id, self.pad_token_id = WORLD.eos_id, WORLD.pad_id
        self.pad_token = "<|endoftext|>"

    def _ids(self, text: str) -> list[int]:
        return list(range(len(text.split()) * WORLD.tokens_per_word))

    def __call__(self, text=None, **kwargs):
        if kwargs:
            raise TypeError(f"fake tokenizer got unexpected kwargs {sorted(kwargs)}")
        return {"input_ids": [self._ids(t) for t in text]}

    def encode(self, text, **kwargs):
        return self._ids(text)


class AutoTokenizer:
    @staticmethod
    def from_pretrained(name, **kwargs):
        bad = sorted(set(kwargs) - _TOKENIZER_KWARGS)
        if bad:
            raise TypeError(f"AutoTokenizer.from_pretrained got unexpected kwargs {bad}")
        WORLD.kwargs_seen["AutoTokenizer"] = {"name": name, **kwargs}
        return FakeTokenizer()
