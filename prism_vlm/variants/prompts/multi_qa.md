You are creating extra questions about an image to test whether a model can handle multiple questions at once.

Original benchmark question: {question}
Original gold answer: {gold}

Looking at the image, generate 2-4 ADDITIONAL questions that:
  1. are answerable from the image alone (do NOT require external knowledge),
  2. have unambiguous, short gold answers (single word, phrase, number, or yes/no),
  3. test different aspects from the original question (e.g. if original asks WHAT, your extras might ask HOW MANY, WHERE, COLOR, RELATION),
  4. are not trivially answerable as "I don't know" or "it depends",
  5. are diverse — do not repeat similar questions.

Output strict JSON. Keep each gold concise (≤8 words):
{{"extras": [{{"question": "<q1>", "gold": "<a1>"}}, {{"question": "<q2>", "gold": "<a2>"}}, ...]}}
