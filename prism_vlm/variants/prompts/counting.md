You are creating a counting question about an image.

Look at the image and produce ONE counting question whose correct answer is an integer between 1 and 100. The objects to count must be:
  1. clearly visible and unambiguous (no partial / occluded / debatable instances),
  2. genuinely countable in the image (do NOT invent objects),
  3. specific enough that the count is exact (e.g. "people wearing red shirts" not just "people"),
  4. not too easy — the count should be at least 3 if possible (avoid trivial 1-2 cases when more are present).

If the image does NOT support a confident integer count, output `null` for question and gold_count.

Output strict JSON:
{{"question": "<the counting question>" or null, "gold_count": <int 1-100> or null, "object": "<what's being counted>"}}
