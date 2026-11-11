You are designing a hallucination trap for a vision-language model. Three sub-types are possible — pick the ONE you'll generate based on `requested_sub_type`.

Image attached. requested_sub_type: {sub_type}

For each sub-type, produce a question that would tempt a weak model to fabricate a confident answer:

- **absent_entity** — Ask about an object, attribute, or relation that is NOT present in the image. Target plausible but genuinely absent entities (no "are there dragons" — too obvious). Vary grammatical form across the dataset (counting, colour, presence, identification, relation). Output extra fields: "absent_entity": "<the missing thing>", "form_category": "counting|colour|presence|identification|relation".

- **misleading_premise** — Embed a FALSE premise about the image inside the question, then ask a follow-up that depends on the premise. Example: image of blue car → "What is the model of the red car in the image?". The correct response is to reject the premise. Output extra field: "false_premise": "<the wrong claim embedded>".

- **uncertain_image** — Ask a specific, committed question that the image does NOT reliably support (because of ambiguity, occlusion, blur, or insufficient detail). Example: a tiny distant face in a crowd → "What is the eye color of the person in the back row?". The correct response is to express uncertainty. Output extra field: "ambiguity_source": "<what makes it unanswerable>".

Output strict JSON:
{{"sub_type": "{sub_type}", "question": "<the trap question>", "<extra_field>": "<value>"}}
