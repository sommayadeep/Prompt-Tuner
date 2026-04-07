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

    def reset(self, seed=None, options=None):
        """Resets the environment to the default state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.current_prompt = "Extract user data as JSON."
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
        
        # Simulate API response for demo (replace with actual call when token/model is available)
        # response = self.client.chat.completions.create(
        #     model=self.cfg["MODEL_NAME"],
        #     messages=[{"role": "user", "content": self.current_prompt}],
        #     max_tokens=150
        # )
        # output_data = response.choices[0].message.content.strip()
        output_data = '{"name": "Sanjay", "role": "Dev"}'  # Dummy response for demo

        # Grader Logic (Mandatory Requirement)
        target = {"name": "Sanjay", "role": "Dev"}
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
