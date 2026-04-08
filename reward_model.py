import json
import re

def grade(output, expected):
    """
    Expert Grader Function for Phase 2 Validation.
    Returns: float strictly BETWEEN 0.0 and 1.0 (never exactly 0.0 or 1.0)
    Handles: keyword extraction, JSON objects, and string matching
    """
    
    # Ensure output is a string
    if not isinstance(output, str):
        output = str(output)
    
    # ✅ CASE 1: Keyword extraction (expected is dict with keywords)
    if isinstance(expected, dict) and "expected_keywords" in expected:
        keywords = expected["expected_keywords"]
        if not keywords:
            return 0.05  # Never 0.0
        
        matched = sum(1 for kw in keywords if str(kw).lower() in output.lower())
        score = (matched / len(keywords)) * 0.9 + 0.05  # Range: [0.05, 0.95]
        return max(0.01, min(0.99, round(score, 3)))
    
    # ✅ CASE 2: Direct keyword list
    if isinstance(expected, list):
        if not expected:
            return 0.05
        matched = sum(1 for kw in expected if str(kw).lower() in output.lower())
        score = (matched / len(expected)) * 0.9 + 0.05
        return max(0.01, min(0.99, round(score, 3)))
    
    # ✅ CASE 3: JSON object matching
    score = 0.0
    try:
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
            data = json.loads(clean_json)
            score += 0.4
            
            if expected and isinstance(expected, dict):
                # Key matching (0.3)
                matching_keys = [k for k in expected.keys() if k in data]
                key_score = (len(matching_keys) / len(expected.keys())) * 0.3
                score += key_score
                
                # Value matching (0.3)
                matching_values = [k for k in matching_keys if str(data[k]) == str(expected[k])]
                val_score = (len(matching_values) / len(expected.keys())) * 0.3
                score += val_score
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    
    # ✅ ENFORCE STRICT BOUNDS: Never exactly 0.0 or 1.0
    return max(0.01, min(0.99, round(score, 3)))

# Local Validation Test
if __name__ == "__main__":
    test_out = '{"name": "Sommayadeep", "role": "Saha"}'
    test_exp = {"name": "Sommayadeep", "role": "Saha"}
    
    final_score = grade(test_out, test_exp)
    print(f"--- [REWARD STATUS] Validation Pass ---")
    print(f"Final Score: {final_score}")
