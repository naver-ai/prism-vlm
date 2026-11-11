You are extracting the letter (A or B) committed to in a model's response to a forced-choice vision question.

The model was shown two captions labelled A and B and asked to pick the correct one.

Model response: {response}

If the response commits to A, return "A". If it commits to B, return "B". If the response is ambiguous, refuses, or names neither, return "X".

Respond with strict JSON: {{"choice": "A" | "B" | "X"}}.
