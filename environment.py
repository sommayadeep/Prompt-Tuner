import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
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
        
        # Lazy initialization or handle missing token for startup
        token = self.cfg.get("HF_TOKEN") or "DUMMY_TOKEN_FOR_STARTUP"
        
        self.client = OpenAI(
            base_url=self.cfg["API_BASE_URL"],
            api_key=token
        )
        
        # Action Space: 5 modifiers
        self.action_space = spaces.Discrete(5)
        # Observation Space: 128-dim simulated state
        self.observation_space = spaces.Box(low=0, high=1, shape=(128,), dtype=np.float32)
        
        self.current_prompt = "Extract user data as JSON."
        self.current_step = 0
        self.max_steps = 3
        self.training_example = {
            "input": "The Eiffel Tower is tall.",
            "expected_keywords": ["Eiffel"],
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to the default state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.current_prompt = "Extract user data as JSON."
        return self._get_obs(), {}

    def _get_obs(self):
        """Simulated observation space (state)."""
        return np.random.rand(128).astype(np.float32)

    def _fallback_chat_completion(self, prompt_text):
        """Fallback HTTP calls for different HF endpoint shapes."""
        base_url = self.cfg["API_BASE_URL"].rstrip("/")
        model = self.cfg["MODEL_NAME"]

        candidate_endpoints = [
            f"{base_url}/chat/completions",
            f"https://router.huggingface.co/v1/chat/completions",
            f"https://router.huggingface.co/hf-inference/models/{model}/v1/chat/completions",
            f"https://api-inference.huggingface.co/v1/chat/completions",
        ]

        headers = {
            "Authorization": f"Bearer {self.cfg.get('HF_TOKEN', '')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 150,
        }

        errors = []
        for endpoint in candidate_endpoints:
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                errors.append(f"{endpoint} -> {type(e).__name__}: {e}")

        raise RuntimeError("All inference endpoints failed: " + " | ".join(errors))

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
        
        # Build task from training data so rewards follow user-provided examples.
        sample_input = self.training_example.get("input", "")
        task_prompt = (
            f"{self.current_prompt}\n"
            "Extract key entities/keywords from the input text and respond in JSON.\n"
            f"Input: {sample_input}\n"
            "Output JSON with a 'keywords' field."
        )

        # Remote inference using OpenAI-compatible client, with HTTP fallback
        try:
            response = self.client.chat.completions.create(
                model=self.cfg["MODEL_NAME"],
                messages=[{"role": "user", "content": task_prompt}],
                max_tokens=150
            )
            output_data = response.choices[0].message.content.strip()
        except Exception:
            output_data = self._fallback_chat_completion(task_prompt)
        # Simulate API response for demo (replace with actual call when token/model is available)
        # output_data = '{"name": "Sanjay", "role": "Dev"}'  # Dummy response for demo

        # Grader Logic (Mandatory Requirement)
        target = self.training_example
        reward = reward_model.grade(output_data, target)
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {"output": output_data}

# Local Integration Test
if __name__ == "__main__":
    env = PromptEnv()
    obs, info = env.reset()
    print(f"--- [ENV STATUS] Initialization Clear ---")
    print(f"Observation Shape: {obs.shape}")
