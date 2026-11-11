You are designing a forced-choice vision-discrimination test. Look at the image attached and produce TWO captions:

1. **true_caption** — A detailed, accurate description of the image (≤60 words). Mention concrete visible details (objects, counts, colours, spatial relations, labels, text).

2. **fake_caption** — Take the true_caption and change EXACTLY ONE specific detail to a plausible-but-wrong value. The change must be:
  - **Visually verifiable**: the swapped detail must be something only checkable by actually looking at the image (not derivable from world knowledge or context).
  - **Plausible to a text-only reader**: the fake should sound natural and not raise suspicion if you only read the words.
  - **Specific and localized**: change a number, colour, position, named object, or text content — not a general claim.

Examples of good swaps:
  - "three apples" → "five apples"
  - "red car" → "blue car"
  - "logo on the left" → "logo on the right"
  - "sign reads 'STOP'" → "sign reads 'YIELD'"

Bad swaps (avoid):
  - Changing the entire scene type ("kitchen" → "beach") — too obvious from rest of caption
  - Changing something not actually visible
  - Adding/removing whole sentences

Output strict JSON:
{{"true_caption": "<60-word detailed description>",
  "fake_caption": "<same text with one visible detail swapped>",
  "swapped_detail": {{"original": "<the true value>", "fake": "<the fake value>"}}}}
