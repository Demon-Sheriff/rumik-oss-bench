"""The transformers 5 CohereTokenizer guard. No model and no torch."""

from __future__ import annotations

import pytest

from rumik_oss_eval.tokenizer_compat import assert_tokenizer_compat, transformers_major


class CohereTokenizer:
    pass


class PreTrainedTokenizerFast:
    pass


class TokenizersBackend:
    pass


def test_major_parses_release_and_rc():
    assert transformers_major("4.57.6") == 4
    assert transformers_major("5.18.0") == 5
    assert transformers_major("5.0.0rc1") == 5


def test_garbage_version_is_rejected():
    with pytest.raises(ValueError):
        transformers_major("v5.1")


def test_transformers_5_cohere_is_refused():
    with pytest.raises(RuntimeError, match="PreTrainedTokenizerFast"):
        assert_tokenizer_compat(CohereTokenizer(), version="5.18.0")


def test_transformers_4_cohere_is_allowed():
    assert_tokenizer_compat(CohereTokenizer(), version="4.57.6") is None


def test_fast_tokenizer_on_5_is_allowed():
    # The config fix names PreTrainedTokenizerFast. On 5.18 the loaded type
    # is TokenizersBackend. Both must pass, or the fix is rejected.
    assert_tokenizer_compat(PreTrainedTokenizerFast(), version="5.18.0") is None
    assert_tokenizer_compat(TokenizersBackend(), version="5.18.0") is None
