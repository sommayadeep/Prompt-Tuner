import os

def get_config():
    """
    Expert Configuration Loader for ML Submission.
    Reads from os.environ as required by the validator.
    """
    config = {
        "API_BASE_URL": os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1"),
        "MODEL_NAME": os.environ.get("MODEL_NAME", "meta-llama/Llama-3-8B-Instruct"),
        "HF_TOKEN": os.environ.get("HF_TOKEN")
    }

    # Strict Validation: Fail fast if critical keys are missing
    missing = [key for key, val in config.items() if not val]
    if missing:
        raise EnvironmentError(
            f"\n[CRITICAL ERROR] Missing mandatory environment variables: {', '.join(missing)}\n"
            "Please ensure these are set in your OS or shell environment before running."
        )

    return config

# Local Check
if __name__ == "__main__":
    try:
        cfg = get_config()
        print("--- [CONFIG STATUS] Ready ---")
        print(f"Base URL: {cfg['API_BASE_URL']}")
        print(f"Model: {cfg['MODEL_NAME']}")
    except EnvironmentError as e:
        print(e)
