"""Load a TinyAya bundle and everything derived from its vocabulary.

The bundle (e.g. `Pranavz/583`) is self-contained: model, expanded tokenizer, and
the Mimi codec all ship together, so nothing is fetched from the gated base repo
at runtime.

Every structural constant is read from `config.json` rather than hardcoded,
because they differ across checkpoint lineages - 583 has vocab 277404 while the
Silver style-FT line has 277406, and a mismatch there produces garbage audio with
no error anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..shapes import AudioVocabLayout
from ..tokenizer_compat import assert_tokenizer_compat
from ..vendor.audio_token_parser import build_audio_code_id_map

DEFAULT_DEVICE = "cuda:0"


@dataclass(slots=True)
class TinyAyaBundle:
    model: Any
    tokenizer: Any
    codec: Any
    feature_extractor: Any

    num_quantizers: int
    codebook_size: int
    audio_start_id: int
    audio_end_id: int
    text_id: int | None
    vocab_size: int
    sample_rate: int

    code_map: dict[int, tuple[int, int]]
    allowed_ids: Any                       # torch.LongTensor on `device`
    layout: AudioVocabLayout
    layout_is_affine: bool

    device: str
    codec_device: str
    speakers: tuple[str, ...] = ()

    def prompt(self, text: str, speaker: str | None = None) -> str:
        """Build the generation prompt.

        Template from the bundle's own inference script:
        `<text>{speaker}: {text}<audio>`. The trailing `<audio>` is what puts the
        model into audio-emission mode; without it you get text continuation.
        """
        body = f"{speaker}: {text}" if speaker else text
        return f"<text>{body}<audio>"


def load_bundle(
    path: str | Path,
    *,
    device: str = DEFAULT_DEVICE,
    codec_device: str | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
) -> TinyAyaBundle:
    """Load model + codec and derive the vocabulary maps.

    `codec_device` defaults to `device`. Decoding on GPU runs ~1800x realtime
    versus ~47x on CPU, which is immaterial for offline eval and very material
    for an RL rollout. Pass "cpu" to fall back.
    """
    import torch
    import transformers
    from transformers import (AutoFeatureExtractor, AutoModelForCausalLM,
                              AutoTokenizer, MimiModel)

    # A Hugging Face repo id ("org/name") is not a filesystem path, and the
    # codec lives in a `codec/` subfolder either way. Detect which we were given
    # so the same call works against the Hub and against a local clone -- the
    # public reproduction path uses the Hub, and a path-only loader would force
    # every user to download and unpack by hand first.
    root = Path(path)
    is_local = root.exists()
    codec_device = codec_device or device
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]

    src = root if is_local else str(path)
    tokenizer = AutoTokenizer.from_pretrained(src)
    # transformers 5 CohereTokenizer retokenizes Indic text and can emit
    # <SEP>/<CLS> past the embedding. Stop before the weight download.
    assert_tokenizer_compat(tokenizer, version=transformers.__version__)
    model = AutoModelForCausalLM.from_pretrained(
        src, trust_remote_code=True, dtype=torch_dtype,
        attn_implementation=attn_implementation,
    ).eval().to(device)

    if is_local:
        codec_dir = root / "codec"
        codec = MimiModel.from_pretrained(
            codec_dir, dtype=torch.float32).eval().to(codec_device)
        feature_extractor = AutoFeatureExtractor.from_pretrained(codec_dir)
    else:
        codec = MimiModel.from_pretrained(
            src, subfolder="codec", dtype=torch.float32).eval().to(codec_device)
        feature_extractor = AutoFeatureExtractor.from_pretrained(src, subfolder="codec")

    cfg = model.config
    vocab = tokenizer.get_vocab()
    code_map = build_audio_code_id_map(vocab)
    if not code_map:
        raise RuntimeError(f"no <code_q> audio tokens found in the tokenizer at {root}")

    num_quantizers = int(getattr(cfg, "num_quantizers"))
    codebook_size = int(getattr(cfg, "codebook_size"))
    audio_start_id = int(getattr(cfg, "audio_start_token_id"))
    audio_end_id = int(getattr(cfg, "audio_end_token_id"))
    embed_rows = int(model.get_input_embeddings().weight.shape[0])

    # The documented failure mode is silent: a tokenizer/embedding mismatch
    # produces plausible-sounding garbage. Assert on the ids we will actually
    # feed rather than on len(tokenizer), which legitimately exceeds the
    # embedding when trailing specials (<SEP>, <CLS>) were appended past it.
    used_max = max(max(code_map), audio_start_id, audio_end_id)
    if used_max >= embed_rows:
        raise RuntimeError(
            f"token id {used_max} exceeds the model embedding ({embed_rows} rows); "
            "tokenizer and checkpoint are misaligned"
        )

    base_id = min(code_map)
    layout = AudioVocabLayout(
        base_id=base_id, num_quantizers=num_quantizers, codebook_size=codebook_size,
        audio_start_id=audio_start_id, audio_end_id=audio_end_id,
    )
    # The closed form (base + code*Q + q) is only valid if no audio token
    # pre-existed in the base vocab; the tokenizer expansion skips duplicates.
    # Verify rather than trust, and record the answer for callers.
    layout_is_affine = layout.verify_against(code_map)

    allowed_ids = torch.tensor(sorted([*code_map, audio_end_id]),
                               dtype=torch.long, device=device)

    return TinyAyaBundle(
        model=model, tokenizer=tokenizer, codec=codec, feature_extractor=feature_extractor,
        num_quantizers=num_quantizers, codebook_size=codebook_size,
        audio_start_id=audio_start_id, audio_end_id=audio_end_id,
        text_id=vocab.get("<text>"), vocab_size=embed_rows,
        sample_rate=int(feature_extractor.sampling_rate),
        code_map=code_map, allowed_ids=allowed_ids,
        layout=layout, layout_is_affine=layout_is_affine,
        device=device, codec_device=codec_device,
        speakers=tuple(getattr(cfg, "speakers", ()) or ()),
    )
