import json
import re

def grade(output, expected):
    """
    Expert Grader Function for ML Submission.
    Returns: float between 0.0 and 1.0.
    Weights: 0.4 (Format) | 0.3 (Keys) | 0.3 (Values)
    """
    score = 0.0
    
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
