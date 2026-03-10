import json

input_path = "aspect_results.jsonl"
output_path = "aspect_results_cleaned.jsonl"

def is_valid_sample(sample: dict) -> bool:
    """Return True only if ALL aspect_terms have non-negative from/to."""
    for aspect in sample.get("aspect_terms", []):
        if aspect.get("from", -1) < 0 or aspect.get("to", -1) < 0:
            return False
    return True

valid_count = 0
invalid_count = 0

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:

    for line_num, line in enumerate(infile, start=1):

        line = line.strip()
        if not line:
            continue

        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            print(f"[WARN] Skipping invalid JSON on line {line_num}")
            invalid_count += 1
            continue

        if is_valid_sample(sample):
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
            valid_count += 1
        else:
            invalid_count += 1

print(f"✔ Done! {valid_count} valid samples written to {output_path}")
print(f"✘ Skipped {invalid_count} invalid samples")
