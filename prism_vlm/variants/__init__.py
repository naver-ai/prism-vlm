"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Per-axis variant definitions.

Each generator turns a source item plus its build-time artifact into the
prompt(s) the evaluee actually sees. ``SUB_TYPES`` is the ``h`` sub-type pool,
which build also needs in order to rotate sub-types per item.
"""

from .base import VariantGenerator
from .counting import Counting
from .h_pool import SUB_TYPES, HPool
from .if_constraints import IFConstraints
from .multi_qa import MultiQA
from .q_original import QOriginal
from .self_correction import SelfCorrection
from .vis_pair import VisPair

__all__ = [
    "SUB_TYPES",
    "Counting",
    "HPool",
    "IFConstraints",
    "MultiQA",
    "QOriginal",
    "SelfCorrection",
    "VariantGenerator",
    "VisPair",
    "all_generators",
]


def all_generators() -> list[VariantGenerator]:
    return [
        QOriginal(),
        IFConstraints(),
        HPool(),
        MultiQA(),
        SelfCorrection(),
        Counting(),
        VisPair(),
    ]
