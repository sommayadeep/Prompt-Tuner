"""
FastAPI Server Application for LLM Prompt Auto-Tuner
Handles /reset and /step endpoints for RL environment
"""

from fastapi import FastAPI, Body, HTTPException
from environment import PromptEnv
import uvicorn

app = FastAPI(
    title="LLM Prompt Auto-Tuner API",
    description="Phase 1 & 2 Compliant RL Environment for Optimizing LLM Prompts",
    version="1.0.0"
)

env = PromptEnv()


@app.post("/reset")
async def reset_env(payload: dict = Body(default={})):
    """
    Mandatory Reset Endpoint (Phase 1 Requirement).
    Resets the environment to initial state.
    """
    options = {}
    if "model_id" in payload:
        options["model_id"] = payload["model_id"]
    if "seed_prompt" in payload:
        options["seed_prompt"] = payload["seed_prompt"]
    if "training_data" in payload:
        options["training_data"] = payload["training_data"]
    
    obs, info = env.reset(options=options)
    return {"observation": obs.tolist(), "info": info}


@app.post("/step")
async def step_env(payload: dict = Body(default={})):
    """
    Mandatory Step Endpoint (Phase 1 Requirement).
    Executes one step in the environment with the given action.
    """
    try:
        # Handle both action formats: int or dict with "command" field
        action = payload.get("action", 0)
        if isinstance(action, dict):
            # Format: {"action": {"command": "..."}}
            action = 0  # Default action when command is passed
        elif isinstance(action, (int, float)):
            # Format: action as integer
            action = int(action)
        else:
            action = 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # ✅ Phase 2: Include grader metadata in response
        return {
            "observation": obs.tolist(),
            "reward": reward,
            "done": terminated or truncated,
            "info": {
                **info,
                "grader": {
                    "name": "reward_model_grade",
                    "score": float(reward),
                    "raw_score": float(reward)
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return {"status": "healthy", "service": "LLM Prompt Auto-Tuner API"}


def main():
    """Main entry point for server deployment (Phase 1 Requirement)."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
