# rumik-oss-bench

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-rumik--ai%2Frumik--oss--1-yellow)](https://huggingface.co/rumik-ai/rumik-oss-1)
[![Tech blog](https://img.shields.io/badge/Blog-rumik.ai%2Fresearch-blue)](https://rumik.ai/research/rumik-oss)
[![License](https://img.shields.io/badge/Code-Apache--2.0-green)](LICENSE)

**[Model](https://huggingface.co/rumik-ai/rumik-oss-1)** ·
**[Tech blog](https://rumik.ai/research/rumik-oss)** ·
**[Methodology](METHODOLOGY.md)** ·
**[Data licences](DATA_LICENSES.md)**

How we measure text-to-speech quality for 21 Indian languages, and how to
reproduce those measurements for any TTS system including ours.

This repo has the prompts, the scoring code, the method, and our per-sample
numbers. It does not have audio: we cannot republish audio the vendor APIs
generated for us, and our own is large and regenerable from the prompts.

## Two things to know before quoting any of this

1. **A gap smaller than 0.006 CER is noise.** That is how much two runs of the
   same checkpoint differ. We mark those "within noise" instead of calling them
   wins. Most of our per-language results are in that band.

2. **On some languages our model scores better than real human speech does
   under this judge.** IndicConformer's median CER on real human Bengali is
   0.0545; ours is 0.0302. Where that happens the number is measuring the judge,
   not the speech, and cannot rank systems.

Both are explained in [METHODOLOGY.md](METHODOLOGY.md). Read it before drawing
conclusions.

## Reproduce our numbers, no model needed

```bash
pip install -e .
pytest -q
```

270 tests, under a second, no GPU and no downloads. Three more modules cover
audio-token parsing and need torch, so they skip unless you install the
`[judges]` extra, 320 in total. They pin the metric definitions, including the
cases where CER implementations silently disagree: does the denominator include
spaces, grapheme clusters or code points, what an empty hypothesis scores.

Then rebuild our tables from the per-sample scores we published:

```bash
python scripts/aggregate.py results/rumik_oss.per_sample.tsv \
    --vendors results/vendor_analysis.json
```

If you get our numbers, our arithmetic is right. If not, tell us.

## Generate our audio and score it

```bash
pip install -e ".[judges]"
```

The model is loaded from [Hugging Face](https://huggingface.co/rumik-ai/rumik-oss-1)
and nowhere else. No weights are vendored into this repo. The Mimi codec ships
inside that model repo, so there is nothing else to fetch.

```bash
python scripts/generate.py --promptset prompts/indic22_bench.jsonl \
    --out audio/rumik_oss

python scripts/score.py --audio audio/rumik_oss \
    --promptset prompts/indic22_bench.jsonl \
    --judge indicconformer --out runs/rumik_oss

python scripts/aggregate.py runs/rumik_oss/per_sample.tsv \
    --vendors results/vendor_analysis.json
```

Generation goes through `rumik_oss_eval.rollout`, the same code path that
produced the published table. Settings default to the published ones:
temperature 0.8, top_k 30, sampling restricted to the audio token ids.

**Verified.** On 2026-09-08 this path was run from a clean clone against
`rumik-ai/rumik-oss-1` and reproduced the published per-prompt scores:

| prompt | published | live | duration ratio |
|---|---|---|---|
| hi_000001 | 0.01205 | 0.01250 | 1.07 |
| hi_000002 | 0.02740 | 0.02516 | 1.19 |
| hi_000003 | 0.02532 | 0.02614 | 1.30 |
| hi_000004 | 0.00000 | 0.00529 | 1.09 |

The residual differences are sampling noise. Generation is stochastic at
temperature 0.8, so exact equality is not expected and would be suspicious.

> **One thing to get right if you write your own generation script.** The model
> is driven with the wire format `<text>{speaker}: {text}<audio>`, and the
> `text` field in `prompts/*.jsonl` is only the middle part. The trailing
> `<audio>` is what puts the model into audio-emission mode. Omit it and the
> model runs long: we measured 1.3x to 2.5x the expected duration, and CER
> charges the trailing speech as insertions. `scripts/generate.py` builds this
> for you via `bundle.prompt()`.

### Stay on transformers 4.x

`[judges]` pins `transformers>=4.57,<5`. That pin is load-bearing.

`rumik-oss-1` sets `tokenizer_class` to `CohereTokenizer`. On transformers 5
that class builds a new pre-tokenizer (`Digits` + `ByteLevel` with its own
regex) and ignores the Split saved in `tokenizer.json`. The saved Split keeps
Unicode combining marks (`\p{M}`) on the letter, which is what holds a matra
or virama on its consonant. Without it the ids change. `decode` still returns
the original string, so nothing warns you.

On `prompts/indic22_bench.jsonl`, transformers 5.18.0 disagrees with 4.57.6
(the `transformers_version` in the model config) on 1715 of 2100 prompts.
Mean length goes from 189.6 tokens to 253.0. `hi_000001` goes from 67 to 159.
Seventeen of the 21 languages differ on every prompt. `ur` differs on 15.
`kn`, `mni`, and `sat` match.

5.x also registers `<SEP>` and `<CLS>` as ids 277404 and 277405. The
embedding has 277404 rows, so a prompt that contains those strings raises
`IndexError` inside the embedding.

`load_bundle` refuses a `CohereTokenizer` on transformers 5 or newer before
it downloads weights. Keep the pin. The other way through is
`"tokenizer_class": "PreTrainedTokenizerFast"` in `tokenizer_config.json`,
which matches 4.57.6 on all 2100 prompts under both 4.57.6 and 5.18.0.

## Score any other system

The judge is IndicConformer, gated on Hugging Face, so request access then
`huggingface-cli login`. Generate audio for the prompts in `prompts/`, one file
per prompt id, then:

```bash
python scripts/score.py --audio audio/my_system \
    --promptset prompts/indic22_bench.jsonl \
    --judge indicconformer --out runs/my_system

python scripts/aggregate.py runs/my_system/per_sample.tsv \
    --vendors results/vendor_analysis.json
```

The output is the same schema as `results/`, so it lines up directly with ours.

## How the vendor numbers were produced

We called each vendor's own API with the same prompts in `prompts/`, following
that vendor's documented prompt format and using their own voices. We did not
try to make the APIs behave like ours: each was driven the way its documentation
says to drive it.

Every vendor was steered toward the same style target, "professional, steady
pace". How you express that differs per vendor, and this is a real limit on any
cross-vendor comparison. Some take words, some take numeric knobs, some take a
category:

| vendor | model id | voice | langs | style channel |
|---|---|---|---|---|
| **rumik-oss (ours)** | `rumik-ai/rumik-oss-1` | Ira | 21 | description tag inside the input text |
| Sarvam Bulbul v3 | `bulbul:v3` | suhani | 10 | numeric parameters (pace, temperature) |
| ElevenLabs v3 | `eleven_v3` | `1Z7Y8o9c...` | 13 | numeric `voice_settings` |
| Smallest.ai Lightning | `lightning_v3.1_pro` | aarini | 10 | numeric parameter (speed) |
| Cartesia Sonic 3.5 | `sonic-3.5` | per-language | 9 | numeric parameters plus a categorical emotion |
| Gemini 3.1 Flash TTS | `gemini-3.1-flash-tts-preview` | Kore | 15 | natural-language preamble in the prompt |
| xAI Grok TTS | unversioned | eve | 1 | numeric parameter (speed) |

Cartesia used a different voice per language. The full mapping, and every field
above, is in `results/vendor_config.json` so it is data, not prose.

Our own model takes style inside the text, so the exact prompt is:

```
<text>Ira: <description="professional, {lang} accent, steady pace"> {text}<audio>
```

The trailing `<audio>` is what puts it into audio mode. Generation is limited to
the 16385 audio token ids. Temperature 0.8, top_k 30.

To reproduce the vendor side you need your own account with each vendor. The
prompts and the exact configuration above are everything else you need.

## What is in here

```
README.md               this file
METHODOLOGY.md          how a system gets a number, and which numbers mean
                        anything. the noise floors are the important part.
DATA_LICENSES.md        where every prompt came from and what its licence is
NOTICE                  attribution for the model, codec and judges
LICENSE                 Apache-2.0, for the code

prompts/
  indic22_bench.jsonl   2100 prompts, 21 languages. the benchmark set behind
                        every number above. generated by Rumik AI.
  heldout_v1.jsonl       180 prompts, 18 languages, never trained on.
                        130 of these are FLEURS (CC BY 4.0, tagged `fleurs`).
  curated_hard_50.jsonl 1050 prompts, 21 languages, deliberately hard.
                        every row stresses digit sequences.

src/rumik_oss_eval/
  metrics/              CER, gCER, WER, length penalty, structural gates
  normalize/            markup strip, number expansion, ISO-15919, phonetic fold
  asr/                  judges: IndicConformer, Whisper, Scribe, script router
  rollout/              the generation path the published numbers came from
  rubric.py             gates plus weighted score composition

scripts/generate.py     generate audio from the Hub model
scripts/score.py        score a directory of audio against a prompt set
scripts/aggregate.py    per-sample scores into the published tables
scripts/check_publishable.py   fails if audio, weights or keys enter the tree

results/                our per-sample scores, vendor scores, vendor config
tests/                  320 tests (270 without torch)
docs/                   the charts
```

> **Note on a similar name.** The vendor files contain a column named
> `silk-oss-v2`. That is an earlier Rumik model the published benchmark
> measured, not the model evaluated here. It keeps its original name because
> those files record what was measured.

Prompt row format, one JSON object per line:

| field | meaning |
|---|---|
| `id` | stable key, also the audio filename and the row key in `results/*.per_sample.tsv` |
| `language` | ISO code, bare. the judge routes on this |
| `speaker` | voice name for our model |
| `text` | full wire-format prompt, description tag included |
| `spoken` | what should be heard. this is the scoring reference |
| `expected_seconds` | peer-median vendor duration, used by the length penalty |
| `tags` | provenance and category |

## The model

Evaluated here: **rumik-oss**, loaded from
[rumik-ai/rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1). That single
checkpoint is the only model evaluated in this repo, and every number above
comes from it.

Per-sample scores for this checkpoint are in
`results/rumik_oss.per_sample.tsv`, and the vendor measurements they are
compared against are in `results/vendor_analysis.json`. Aggregate them yourself
with `scripts/aggregate.py`; the noise band described above applies to every
per-language margin you will get out of it.

More detail in the [tech blog](https://rumik.ai/research/rumik-oss).

## Licence and attribution

The code in this repo is Apache-2.0 (see [LICENSE](LICENSE)).

The model is not. rumik-oss is a fine-tune of
[CohereLabs/tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire),
which is **CC BY-NC 4.0**: attribution required, and commercial use prohibited.
Those conditions carry to derivatives. The Apache-2.0 grant here covers this
code and does not extend to the model weights.

Other components we depend on and thank:

| component | by | licence |
|---|---|---|
| [Mimi codec](https://huggingface.co/kyutai/mimi) | Kyutai | CC BY 4.0 |
| [IndicConformer 600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | AI4Bharat | MIT (access-gated on HF) |
| [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | OpenAI | MIT |
| [tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire) | Cohere Labs | CC BY-NC 4.0 |

The prompts are ours: `indic22_bench` and `curated_hard_50` were generated by
Rumik AI for this benchmark. The tags on `indic22_bench` (`wiki`, `book`,
`news`, `conv`, `alexa`, `bb`, `umang`, `proper_noun`) are register labels, the
style a prompt is written in, not sources.

The exception is 130 rows in `heldout_v1.jsonl`, which come from FLEURS and are
licensed CC BY 4.0:

> FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech.
> Conneau, Ma, Khanuja, Zhang, Axelrod, Dalmia, Riesa, Rivera, Bapna.
> IEEE SLT, 2023.

Those rows carry the `fleurs` tag, so you can filter them out if you need a set
free of third-party material. Full detail in
[DATA_LICENSES.md](DATA_LICENSES.md).

Full attribution is in [NOTICE](NOTICE).

Vendor names are used only to report measurements of publicly available
services. No affiliation or endorsement is implied, and no vendor-generated
audio is redistributed here.
