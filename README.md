# PRISM-VLM: A Multi-Axis Discriminative Benchmark for Compact Vision-Language Models

**PRISM-VLM** scores every item along **seven axes** — task quality, instruction
following, hallucination resistance, multi-question load, resistance to
conversational pressure, counting, and visual discrimination — and combines them
into a single **PScore** by harmonic mean. Items are recycled from fifteen public
benchmarks and then perturbed per item by a generation judge, which restores
discrimination at the compact tier where single-accuracy suites have saturated.

[![Conference](https://img.shields.io/badge/EMNLP-2026%20Findings-red)]()
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> This repository builds the benchmark, runs models on it, and scores them.
> The cross-judge audit and the paper's analyses are separate stages and are
> not included yet.

## Introduction

<p align="center">
  <img src="docs/axes.svg" width="880" alt="The seven PRISM-VLM axes: q, if, h, mq, sc, c and vis, each with its perturbation, its pass/fail rule, and the build artifact it carries.">
</p>

<p align="center">
  <em>The seven axes. <b>q</b> is task quality; <b>h</b>, <b>sc</b> and <b>vis</b> probe
  behavioral robustness; <b>if</b>, <b>mq</b> and <b>c</b> probe capability bottlenecks.
  Five of the seven carry a per-item artifact written by <code>prism-vlm build</code>.</em>
</p>

- **15** public benchmarks across **5** content domains, sampled at 100 items
  per benchmark (`sampling.items_per_benchmark`) for each of R=5 seeds and
  deduplicated within a benchmark: **6,238 rows**. Counted as distinct source
  items the pool is **6,213** — the BLINK-Counting and BLINK rows share 25
  items, which carry independently synthesised perturbations under each row.
- **Five axes are judge-built** and carry a per-item artifact: the trap question
  (`h`), the bundled extra questions (`mq`), the counting question (`c`), the
  caption pair (`vis`), and the false answer planted at turn Q3 (`sc`). **`q`**
  reuses the recycled question and gold unchanged, and **`if`** samples 1–3
  formatting constraints deterministically from a fixed pool at run time.
- The harmonic mean is deliberate: a single weak axis pulls PScore down instead
  of being averaged away, which is what makes the aggregate least recoverable
  from single-prompt accuracy alone.
- In the paper, PScore separates **97.0%** of the 861 pairs among 42 compact
  models under an item-level paired bootstrap, versus **44–80%** for any single
  prior benchmark.

The pipeline has three stages: **build** synthesises the benchmark once,
**run** evaluates each model against it and grades every axis, and **score**
aggregates the graded items into per-axis means and PScore.

## Software Installation

- Requires-Python >=3.11

```bash
git clone https://github.com/naver-ai/prism-vlm.git
cd prism-vlm
pip install -e .
```

Evaluating a Claude or Gemini model needs an extra:

```bash
pip install -e ".[anthropic]"   # or ".[google]", or ".[all]"
```

Then copy `.env.example` to `.env` and fill it in. `OPENAI_API_KEY` is always
required — the generation and grading judge defaults to `gpt-5`. `HF_TOKEN` is
needed only if one of the source benchmarks is gated on the HuggingFace Hub.

## Source Data

The benchmark items come from fifteen public benchmarks. This repository does
not include them. Fourteen download from the HuggingFace Hub on your machine.
Each source has its own terms; NOTICE lists them, as published on 2026-10-01,
for reference only. The original authors' current terms take precedence, and
you are responsible for complying with them.

The first `build` or `run` prints these terms and asks you to accept them.
Pass `--accept-dataset-terms` to accept them without the prompt, e.g. in a
batch job. The acceptance is recorded once per cache.

Some sources (ScienceQA, part of MMStar, and the Flickr-image benchmarks GQA,
OK-VQA, RefCOCOg and RefCOCO+) do not permit commercial use. Do not use the
results on them for commercial purposes.

### NLVR2

The authors of NLVR2 do not distribute its images publicly. Get them from the
authors directly:

1. Fill in the Google Form linked under *Direct image download* in the NLVR2
   README (https://github.com/lil-lab/nlvr/tree/master/nlvr2) and accept the
   authors' terms of service.
2. Unpack the test1 images they send into one directory, e.g. `~/nlvr2`. The
   files are named `test1-<set>-<pair>-img0.png` and `-img1.png`.
3. Put `nlvr2/data/balanced/balanced_test1.json` from the same repository in
   that directory.
4. Set `PRISM_NLVR2_DIR=~/nlvr2` in `.env`.

Without `PRISM_NLVR2_DIR`, `build` skips NLVR2 and prints a message. The pool
then has fourteen benchmarks, and its scores are not comparable with the
paper's.

## Benchmark Generation Guide

Start with the smoke config: one item per benchmark, one seed, so 15 items and
75 artifacts. It exercises every loader and every generation prompt at roughly
1/500 of the cost.

```bash
prism-vlm build --config configs/smoke_run.yaml
```

```bash
[build] generating bundles for 15 items (judge=gpt-5 effort=low)
[build] done — h:15 mq:15 c:15 vis:15 sc:15 / 15
```

The full build is the same command against the full config:

```bash
prism-vlm build --config configs/main_run.yaml
```

Even the smoke config downloads all fourteen Hub benchmarks. To exercise the
pipeline against one or two of them instead, narrow the pool — `build` and
`run` take the same three flags, and you should pass them the same way to both:

```bash
prism-vlm build --config configs/smoke_run.yaml --benchmarks mmstar okvqa
prism-vlm run   --config configs/smoke_run.yaml --benchmarks mmstar okvqa --models qwen3-vl-2b_off
prism-vlm score --config configs/smoke_run.yaml
```

- `--benchmarks ID...` and `--domains NAME...` restrict which source benchmarks
  are sampled at all, so only those datasets are fetched.
- `--limit N` caps the pool at N items. The cap takes the lowest item ids
  rather than the first N the sampler happens to yield, so build and run land
  on the same items.
- Items the build skipped carry no artifact, so a `run` narrowed differently
  from its `build` scores them 0 on `h`, `mq` and `c` — see the Scoring Guide.

- The first run downloads the fourteen Hub benchmarks into `cache/hf/`, which
  dominates both the wall clock and the disk footprint.
- Synthesis is ~31k judge calls (5 artifacts × 6,238 items).
- Builds resume. Every artifact is a separate file and every step checks for it
  before calling the judge, so an interrupted build can simply be re-run and a
  finished build re-run is a no-op.
- `manifest.json` reports the per-axis acceptance rate. Items the judge declined
  — no confident integer count, a degenerate caption pair — carry no artifact and
  are never asked on `h`, `mq`, `c` and `vis`. On `vis` they leave that axis's
  mean; on `h`, `mq` and `c` they score 0. `sc` is the exception: it still runs,
  with a generic "something different" in place of the planted answer, so the
  item is scored on a weaker pressure than its neighbours. The same items are
  declined for every model either way.

Two things are pinned so a rebuild lands on the same pool. Item **selection**
uses a seeded NumPy generator over the seeds `{0, 1, 42, 1234, 2026}`. Item
**identity** is `sha1(benchmark|orig_index|question)[:16]`, so overlapping draws
across seeds collapse onto one cache entry, and the `h` sub-type rotation is
derived from that id rather than from Python's per-process-randomised `hash`.
The judge is not pinned: artifacts are model outputs -- at the model's own
default temperature for the shipped `gpt-5`, which rejects any other -- and a
different snapshot of it will word them differently.

## Evaluation Guide

Declare the models you want to score in [configs/models.yaml](configs/models.yaml).
Three entries ship as a runnable subset, one per provider pattern:

| Provider | Example | What it needs |
|---|---|---|
| `vllm` | `qwen3-vl-2b_off` | your own endpoint (below) |
| `openrouter` | `qwen3-vl-8b_off` | `OPENROUTER_API_KEY` |
| `openai` | `gpt-5-mini-2025-08-07_min` | `OPENAI_API_KEY` |

vLLM endpoints are deployment-specific, so they live outside the registry. Copy
[configs/endpoints.example.yaml](configs/endpoints.example.yaml) to
`configs/endpoints.yaml` (git-ignored) and point each model at your server, or
export `PRISM_VLLM_ENDPOINT` for a single-server setup.

```bash
prism-vlm run --config configs/smoke_run.yaml --models qwen3-vl-2b_off
```

```bash
[run] qwen3-vl-2b_off: provider=vllm concurrency=16
[run] qwen3-vl-2b_off: hydrating 15 items (15 pending simple-axis, 0 sc-only)
[run] qwen3-vl-2b_off: hydration done, starting inference
[run] qwen3-vl-2b_off: sc dialogues × 15
[run] qwen3-vl-2b_off: judging × 15
[run] qwen3-vl-2b_off: judged 1/15 (6%)
...
[run] qwen3-vl-2b_off: judged 15/15 (100%)
```

- `--models` selects a subset of the registry; omit it to run all of them.
- `--benchmarks`, `--domains` and `--limit` narrow the item pool exactly as
  they do for `build` — see above.
- `--axes` restricts inference to some of `q IF h mq sc c vis`, which is useful
  when re-running one axis after a serving failure.
- Runs resume like builds: every model call and every judge verdict is cached
  per item under `cache/runs/<run_id>/`.
- Calls run at temperature 0 wherever the provider accepts it. Two exceptions
  run at the model's own default instead: the GPT-5 / o-series models, which
  reject a non-default temperature and which the default `gpt-5` judge is one
  of, and every Anthropic model, since Claude 4.7 and later reject the
  parameter outright and the client does not try to tell the generations
  apart. Every call gets the same output budget; raise it for one model with
  `max_tokens` in [configs/models.yaml](configs/models.yaml).
- The `sc` dialogue stops at the first flip, so a model that breaks at Q1 is
  never shown the planted false answer.

Per-item judgments, model responses, and the leaderboard are not distributed
from this repository.

## Scoring Guide

`run` leaves one row per item under `cache/runs/<run_id>/scores/`. `score`
turns those into the table:

```bash
prism-vlm score --config configs/smoke_run.yaml
```

```bash
aggregator: harmonic

model                           n      q     IF      h     mq     sc      c    vis   PScore
-------------------------------------------------------------------------------------------
gpt-5-mini-2025-08-07_min      15  0.733  0.578  1.000  0.733  1.000  0.733  1.000    0.794
qwen3-vl-2b_off                15  0.400  0.133  0.400  0.000  1.000  0.400  1.000    0.000
```

The numbers above come from a 15-item smoke run and say nothing about either
model; this repository publishes no leaderboard. Note the second row: `mq` is
0, so the harmonic mean collapses whatever the other six axes did.

- Every model found in the run is reported; `--models` narrows that.
- `--output DIR` also writes `scores.json` and `scores.csv`.
- `--aggregator {harmonic,geometric,arithmetic,min}` changes how the seven axis
  means combine. **PScore is the harmonic mean** — the other three are there to
  reproduce the aggregator comparison in the appendix, not to replace it.

Two rules decide the numbers, and both differ from a naive column average:

- **`sc` is conditional.** A model that was already wrong at turn 0 has no
  answer to hold on to, so resistance is measured only over the items whose
  `q` verdict is 1.
- **A declined item leaves the mean only on `vis`.** An item the build judge
  refused was never asked on that axis, but only `vis` records that as "no
  value"; `h`, `mq` and `c` record a 0 and count it as a failure. The published
  numbers were computed that way, so `score` reproduces it. The same items are
  declined for every model, so this shifts every model's mean alike rather
  than reordering them.

PScore aggregates the seven per-axis means rather than the per-item rows, and
each mean is floored at 1e-6 so an empty axis gives a near-zero PScore instead
of a division by zero.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Offline and key-free: the build loop runs against a stub judge, the run
stage's graders against a stub client, and the rest pin the determinism
guarantees above.

## How to Cite

```bibtex
@inproceedings{park2026prismvlm,
  title     = {{PRISM-VLM}: A Multi-Axis Discriminative Benchmark for Compact Vision-Language Models},
  author    = {Park, Sanghee and Kim, Kee-Eung},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## License

Licensed under the Apache License, Version 2.0 (Apache-2.0). See
[LICENSE](LICENSE) and [NOTICE](NOTICE). The license covers the source code; the
prompt templates under `prism_vlm/variants/prompts/` and
`prism_vlm/evaluation/judge_prompts/` are released under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).

Source benchmarks are not redistributed in this repository; `prism-vlm build`
fetches them at run time (NLVR2 from your own copy, see Source Data), and each
stays under its own license and terms of use (see NOTICE → Data provenance).
