"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

The terms each source benchmark's original authors publish, as shown to the
user before the first download. NOTICE carries the same table; a test keeps
the two in step.

This is a summary for reference, not a warranty: the original authors' current
terms take precedence, and the user is responsible for complying with them.
"""

from __future__ import annotations

from pathlib import Path

TERMS_AS_OF = "2026-10-01"

# id -> (benchmark, original source, terms as the original authors state them)
SOURCES: dict[str, tuple[str, str, str]] = {
    "mmstar": ("MMStar", "https://github.com/MMStar-Benchmark/MMStar",
               "no license stated; items inherit the terms of the benchmark each "
               "was drawn from, some of which are non-commercial (SEED-Bench, ScienceQA)"),
    "gqa": ("GQA", "https://cs.stanford.edu/people/dorarad/gqa/",
            "annotations CC BY 4.0; images via Visual Genome / MS COCO from Flickr, "
            "no commercial use"),
    "okvqa": ("OK-VQA", "https://okvqa.allenai.org/",
              "no license stated; images from MS COCO (Flickr), no commercial use"),
    "mmmu": ("MMMU", "https://mmmu-benchmark.github.io/", "Apache-2.0"),
    "mathvista": ("MathVista", "https://mathvista.github.io/",
                  "CC BY-SA 4.0 for new contributions; source images and questions "
                  "stay with their original authors"),
    "scienceqa": ("ScienceQA", "https://scienceqa.github.io/",
                  "CC BY-NC-SA 4.0, non-commercial research only"),
    "ai2d": ("AI2D", "https://allenai.org/data/diagrams", "CC BY-SA 4.0"),
    "chartqa": ("ChartQA", "https://github.com/vis-nlp/ChartQA",
                "GPL-3.0; charts crawled from Statista, Pew Research, Our World in "
                "Data and OECD, whose own terms also apply"),
    "docvqa": ("DocVQA", "https://www.docvqa.org/",
               "no license granted; images from the UCSF Industry Documents Library, "
               "whose rights holders keep their copyright"),
    "refcocog": ("RefCOCOg", "https://github.com/lichengunc/refer",
                 "no license stated for the annotations; images from MS COCO (Flickr), "
                 "no commercial use"),
    "refcocoplus": ("RefCOCO+", "https://github.com/lichengunc/refer",
                    "no license stated for the annotations; images from MS COCO "
                    "(Flickr), no commercial use"),
    "tallyqa": ("BLINK Counting", "https://zeyofu.github.io/blink/", "Apache-2.0"),
    "mantis": ("Mantis-Eval", "https://tiger-ai-lab.github.io/Mantis/", "Apache-2.0"),
    "mmt_bench": ("NLVR2", "https://lil.nlp.cornell.edu/nlvr/",
                  "images are not publicly distributed; request them from the "
                  "original authors and accept their terms (research use only)"),
    "blink": ("BLINK", "https://zeyofu.github.io/blink/", "Apache-2.0"),
}


def terms_text(benchmark_ids: list[str]) -> str:
    lines = [
        "PRISM-VLM downloads the source benchmarks below. Their terms, as the",
        f"original authors published them on {TERMS_AS_OF}, are summarised for",
        "reference only and are not warranted. The original authors' current terms",
        "take precedence, and you are responsible for complying with them.",
        "Results on non-commercial sources must not be used for commercial purposes.",
        "",
    ]
    for bid in benchmark_ids:
        name, url, terms = SOURCES[bid]
        lines.append(f"  {name} ({url})")
        lines.append(f"      {terms}")
    lines += ["", "See NOTICE -> Data provenance for the full table."]
    return "\n".join(lines)


def acceptance_marker(cache_root: Path) -> Path:
    """Recorded once per cache and per terms date, so a later revision asks again."""
    return Path(cache_root) / f".dataset_terms_accepted_{TERMS_AS_OF}"
