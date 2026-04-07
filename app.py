from fastapi import FastAPI, Body, HTTPException
from environment import PromptEnv
import uvicorn

app = FastAPI(title="LLM Prompt Auto-Tuner Submission API")
env = PromptEnv()

@app.post("/reset")
async def reset_env(payload: dict = Body(default={})):
    """Mandatory Reset Endpoint."""
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
async def step_env(action: int = Body(..., embed=True)):
    """Mandatory Step Endpoint."""
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        return {
            "observation": obs.tolist(),
            "reward": reward,
            "done": terminated or truncated,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import gradio as gr
from ui import demo
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    # Validator expects port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
