"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

PRISM-VLM: a multi-axis discriminative benchmark for compact VLMs.

``prism-vlm build`` samples source items from the public benchmark pool and
synthesises the per-item artifacts for the five judge-built axes (``h``,
``mq``, ``c``, ``vis``, ``sc``); ``prism-vlm run`` evaluates a model on the
seven axes and grades each; ``prism-vlm score`` aggregates the graded items
into per-axis means and PScore.
"""

__version__ = "0.1.0"
