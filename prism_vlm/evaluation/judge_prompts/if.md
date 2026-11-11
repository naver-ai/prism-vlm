You are grading whether a model's response complied with a list of formatting and instruction-following constraints.

The user gave the model these constraints (numbered):
{constraints}

The model's response was:
---
{response}
---

For EACH numbered constraint, decide if the response satisfies it. Be strict — partial compliance counts as 0. Ignore the answer's correctness; only judge constraint compliance.

Respond with strict JSON in this exact shape, with one entry per constraint in order:
{{"verdicts": [0_or_1, 0_or_1, ...]}}
