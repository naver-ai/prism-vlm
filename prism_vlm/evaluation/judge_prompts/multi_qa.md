You are grading a model's multi-question answer. The user asked the model {n} questions about the same image in a single prompt; the model produced one combined response containing all answers.

Questions and gold answers (numbered):
{qa_block}

Model's combined response:
---
{response}
---

For EACH numbered question, decide whether the model's corresponding answer is semantically correct. Be lenient about formatting (punctuation, casing, article words, synonyms) but strict about content.

If the model failed to answer a question (skipped it or gave an unrelated answer), verdict 0 for that question.

Respond with strict JSON in this exact shape, with one entry per question in order:
{{"verdicts": [0_or_1, 0_or_1, ...]}}
