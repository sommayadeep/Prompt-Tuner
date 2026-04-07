import gymnasium as gym
from gymnasium import spaces
import numpy as np
from openai import OpenAI
import config
import reward_model

class PromptEnv(gym.Env):
    """
    Expert Gymnasium Environment for Prompt Optimization.
    Compliant with the 'OpenEnv' standard for submission validation.
    """
    def __init__(self):
        super(PromptEnv, self).__init__()
        self.cfg = config.get_config()
        self.client = OpenAI(
            base_url=self.cfg["API_BASE_URL"],
            api_key=self.cfg["HF_TOKEN"]
        )
        try:
            from huggingface_hub import InferenceClient
            self.hf_client = InferenceClient(token=self.cfg["HF_TOKEN"])
        except ImportError:
            self.hf_client = None
        
        # Action Space: 5 modifiers
        self.action_space = spaces.Discrete(5)
        # Observation Space: 128-dim simulated state
        self.observation_space = spaces.Box(low=0, high=1, shape=(128,), dtype=np.float32)
        
        self.current_prompt = "Extract user data as JSON."
        self.current_step = 0
        self.max_steps = 5

    def reset(self, seed=None, options=None):
        """Resets the environment to the default state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.current_prompt = "Extract user data as JSON."
        self.target = {"name": "Sanjay", "role": "Dev"}
        self.input_text = ""
        
        if options:
            if "model_id" in options:
                self.cfg["MODEL_NAME"] = options["model_id"]
            if "seed_prompt" in options:
                self.current_prompt = options["seed_prompt"]
            if "training_data" in options and len(options["training_data"]) > 0:
                td = options["training_data"][0]
                self.input_text = td.get("input", "")
                if "expected_keywords" in td:
                    self.target = td["expected_keywords"]
                elif "target" in td:
                    self.target = td["target"]
                    
        return self._get_obs(), {}

    def _get_obs(self):
        """Simulated observation space (state)."""
        return np.random.rand(128).astype(np.float32)

    def step(self, action):
        """
        Applies a prompt modifier, calls the LLM, and calculates the reward.
        """
        self.current_step += 1
        
        modifiers = [
            "Be concise.",
            "Use valid JSON.",
            "Explain keys.",
            "Include 'id'.",
            "Focus strictly on data extraction."
        ]
        
        # Apply Modification
        mod = modifiers[action]
        self.current_prompt = f"{self.current_prompt}. {mod}"
        
        # Build query
        query = f"{self.current_prompt}\n\nInput: {self.input_text}" if getattr(self, "input_text", "") else self.current_prompt
        
        # Remote Inference
        try:
            model_id = self.cfg["MODEL_NAME"]
            if model_id == "meta-llama/Llama-3-8B-Instruct":
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # the exact path required by hub

            if self.hf_client:
                # Use official robust HuggingFace python client
                response = self.hf_client.chat_completion(
                    messages=[{"role": "user", "content": query}],
                    model=model_id,
                    max_tokens=150
                )
                output_data = response.choices[0].message.content.strip()
            else:
                # Fallback to pure openai client interface
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": query}],
                    max_tokens=150
                )
                output_data = response.choices[0].message.content.strip()
        except Exception as e:
            output_data = f"ERROR - {str(e)}"

        # Grader Logic (Mandatory Requirement)
        target = getattr(self, "target", {})
        reward = reward_model.grade(output_data, target)
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {"output": output_data, "prompt": self.current_prompt}

# Local Integration Test
if __name__ == "__main__":
    env = PromptEnv()
    obs, info = env.reset()
    print(f"--- [ENV STATUS] Initialization Clear ---")
    print(f"Observation Shape: {obs.shape}")
