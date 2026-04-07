import json
import re


def _normalize_text(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _extract_predicted_keywords(output):
    """Extract keyword candidates from JSON output when possible."""
    try:
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        clean_json = json_match.group(0) if json_match else output
        data = json.loads(clean_json)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []

    # Prefer explicit keyword fields when present.
    for key in ["keywords", "expected_keywords", "entities", "tags"]:
        if key in data:
            val = data[key]
            if isinstance(val, list):
                return [_normalize_text(v) for v in val]
            return [_normalize_text(val)]

    # Fallback: flatten all leaf string/int values from JSON object.
    values = []
    for v in data.values() if isinstance(data, dict) else []:
        if isinstance(v, (str, int, float)):
            values.append(_normalize_text(v))
        elif isinstance(v, list):
            values.extend(_normalize_text(item) for item in v)
    return values


def _count_fuzzy_matches(expected_items, predicted_items):
    """Count one-to-one fuzzy matches (exact, substring, or superstring)."""
    used_pred_indices = set()
    match_count = 0

    for exp in expected_items:
        for idx, pred in enumerate(predicted_items):
            if idx in used_pred_indices:
                continue
            if exp == pred or exp in pred or pred in exp:
                used_pred_indices.add(idx)
                match_count += 1
                break

    return match_count

def grade(output, expected):
    """
    Expert Grader Function for ML Submission.
    Returns: float between 0.0 and 1.0.
    Weights: 0.4 (Format) | 0.3 (Keys) | 0.3 (Values)
    """
    score = 0.0
    
    # Keyword-based grading path for dataset style examples.
    if expected and isinstance(expected, dict) and "expected_keywords" in expected:
        keywords = [_normalize_text(k) for k in expected.get("expected_keywords", [])]
        if not keywords:
            return 0.0

        predicted = _extract_predicted_keywords(output)
        if not predicted:
            # If output is not JSON, fall back to weak text matching with cap.
            output_lc = _normalize_text(output)
            hits = sum(1 for kw in keywords if kw in output_lc)
            return round(min((hits / len(keywords)) * 0.6, 0.6), 2)

        expected_items = sorted(set(keywords))
        predicted_items = sorted(set(predicted))
        match_count = _count_fuzzy_matches(expected_items, predicted_items)

        precision = match_count / len(predicted_items) if predicted_items else 0.0
        recall = match_count / len(expected_items) if expected_items else 0.0

        if precision + recall == 0:
            return 0.0

        # F1-like score rewards both completeness and avoiding extra wrong keywords.
        f1 = (2 * precision * recall) / (precision + recall)
        return round(min(max(f1, 0.0), 1.0), 2)

    # 1. Format Check (0.4)
    # Attempt to extract JSON if the model is talkative
    try:
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        clean_json = json_match.group(0) if json_match else output
        data = json.loads(clean_json)
        score += 0.4
        
        if expected and isinstance(expected, dict):
            # 2. Key Check (0.3)
            matching_keys = [k for k in expected.keys() if k in data]
            key_score = (len(matching_keys) / len(expected.keys())) * 0.3
            score += key_score
            
            # 3. Value Check (0.3)
            matching_values = [k for k in matching_keys if str(data[k]) == str(expected[k])]
            val_score = (len(matching_values) / len(expected.keys())) * 0.3
            score += val_score

    except (json.JSONDecodeError, AttributeError):
        # Invalid format results in 0.0 for this block
        pass

    return round(min(score, 1.0), 2)

# Local Validation Test
if __name__ == "__main__":
    test_out = '{"name": "Sanjay", "role": "Dev"}'
    test_exp = {"name": "Sanjay", "role": "Dev"}
    
    final_score = grade(test_out, test_exp)
    print(f"--- [REWARD STATUS] Validation Pass ---")
    print(f"Final Score: {final_score}")
