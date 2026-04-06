import sys
from openai import OpenAI
import config
import reward_model

def run_inference():
    """
    Expert Inference Master Script.
    Fulfills Phase 6 'CRITICAL' logging requirements.
    """
    try:
        cfg = config.get_config()
        client = OpenAI(base_url=cfg["API_BASE_URL"], api_key=cfg["HF_TOKEN"])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    tasks = [
        {"name": "json_extraction", "input": "Extract name and age: Sanjay is 25.", "target": {"name": "Sanjay", "age": 25}},
        {"name": "key_value_pairs", "input": "Extract job and city: I am a coder from NYC.", "target": {"job": "coder", "city": "NYC"}},
        {"name": "classification", "input": "Is this positive? 'I love this!'", "target": {"sentiment": "positive"}}
    ]

    total_score = 0.0

    print("[START]")
    for task in tasks:
        print(f"task: {task['name']}")
        
        # Expert Prompt Strategy
        system_prompt = "You are a data extraction assistant. Respond ONLY with valid JSON."
        full_prompt = f"{system_prompt}\n\nInput: {task['input']}"
        
        # [STEP] START
        print("\n[STEP]")
        print(f"input: {task['input']}")
        print(f"prompt: {full_prompt}")
        
        try:
            response = client.chat.completions.create(
                model=cfg["MODEL_NAME"],
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=150
            )
            output = response.choices[0].message.content.strip()
            reward = reward_model.grade(output, task["target"])
            
            print(f"output: {output}")
            print(f"reward: {reward}")
            total_score += reward

        except Exception as e:
            print(f"output: ERROR - {str(e)}")
            print("reward: 0.0")

    # [END] START
    print("\n[END]")
    avg_score = total_score / len(tasks)
    print(f"score: {avg_score:.2f}")

if __name__ == "__main__":
    run_inference()
