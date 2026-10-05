"""Refuse transformers 5's CohereTokenizer before rumik-oss-1 weights download.

transformers 5.x ``CohereTokenizer.__init__`` replaces the pre-tokenizer from
``tokenizer.json`` with ``Digits(individual_digits=True)`` plus ``ByteLevel``
(``use_regex=True`` by default). rumik-oss-1's saved pre-tokenizer is a Split
whose letter classes include ``\\p{M}`` (combining marks), then ByteLevel with
``use_regex`` false. Dropping that Split changes ids on 1715 of 2100
``indic22_bench`` prompts versus transformers 4.57.6, which is the
``transformers_version`` in the model config. ``decode`` still returns the
original text, so the drift is silent. The 5.x class also appends ``<SEP>``
and ``<CLS>`` as ids 277404 and 277405, one past ``vocab_size``.

The ``[judges]`` extra pins ``transformers>=4.57,<5``. The other fix is
``"tokenizer_class": "PreTrainedTokenizerFast"`` in ``tokenizer_config.json``.
On 5.18 that loads as ``TokenizersBackend``, which this check allows.
"""

from __future__ import annotations


def transformers_major(version: str) -> int:
    head = version.split(".", 1)[0]
    if not head.isdigit():
        raise ValueError(f"unrecognized transformers version {version!r}")
    return int(head)


def assert_tokenizer_compat(tokenizer: object, *, version: str) -> None:
    """Raise if this tokenizer would silently disagree with transformers 4.57.6."""
    if transformers_major(version) < 5:
        return
    if type(tokenizer).__name__ != "CohereTokenizer":
        return
    raise RuntimeError(
        f"transformers {version} CohereTokenizer rebuilds its pre-tokenizer "
        "(Digits + ByteLevel) and ignores the Split regex in tokenizer.json. "
        "On rumik-oss-1 that splits Indic combining marks: 1715/2100 indic22 "
        "prompts tokenize differently than on transformers 4.57.6, and "
        "<SEP>/<CLS> are added past the 277404-row embedding. "
        "Pin transformers<5 (the [judges] extra does) or set tokenizer_config.json "
        'tokenizer_class to "PreTrainedTokenizerFast".'
    )
