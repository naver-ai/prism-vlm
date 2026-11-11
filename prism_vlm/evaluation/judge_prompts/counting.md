You are grading a vision-language model's answer to a counting question.

Question: {question}
Correct count: {gold_count}
Model response: {response}

Extract the count the model committed to in its response (it may include reasoning before the final number — only the committed final count matters). The verdict is 1 if and only if the extracted count exactly equals {gold_count}; otherwise 0. Off-by-one or "approximately" does not count.

If the model refused or gave no number, verdict 0.

Respond with strict JSON: {{"verdict": 0 | 1, "extracted": <int_or_null>, "reason": "<one sentence>"}}.
