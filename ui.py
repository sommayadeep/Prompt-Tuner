import gradio as gr
import json
import os
import re
import ast
from environment import PromptEnv

# Define the custom Theme
ocean_theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

# Helpers
def _relaxed_json_loads(text):
    """
    Accepts standard JSON, or JSON with trailing commas / extra whitespace,
    and even Python-literal style as a last resort. Falls back to raising
    the original JSON error so the UI can surface it cleanly.
    """
    if text is None or not str(text).strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Strip trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\\]])", r"\\1", text)
        try:
            return json.loads(cleaned)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception:
                raise e

# API Helper function
def run_optimization(model_id, seed_prompt, training_data):
    # Direct calls to the environment for demonstration
    yield "Starting Optimization Process...", "", "Initializing Environment..."
    
    try:
        # Validate JSON input early so users get immediate feedback.
        parsed_data = _relaxed_json_loads(training_data) or []

        env = PromptEnv()
        if model_id:
            env.cfg["MODEL_NAME"] = model_id.strip()
        if isinstance(parsed_data, list) and parsed_data:
            if isinstance(parsed_data[0], dict):
                env.training_example = parsed_data[0]
        elif isinstance(parsed_data, dict):
            env.training_example = parsed_data

        obs, info = env.reset()
        if seed_prompt:
            env.current_prompt = seed_prompt.strip()

        yield "Environment Reset Successful", "", "Running multi-step optimization..."

        best_reward = float("-inf")
        best_prompt = env.current_prompt
        logs = []

        for step_idx in range(env.max_steps):
            action = step_idx % env.action_space.n
            obs, reward, terminated, truncated, info = env.step(action)
            short_output = str(info.get("output", ""))[:120].replace("\n", " ")
            logs.append(f"Step {step_idx + 1}: Action {action} -> Reward {reward} | Output: {short_output}")

            if reward > best_reward:
                best_reward = reward
                best_prompt = env.current_prompt

            yield best_prompt, str(best_reward), "\n".join(logs)

            if terminated or truncated:
                break
            
    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"
        yield "Error", "Error", error_text


with gr.Blocks(theme=ocean_theme, title="Prompt Auto-Tuner Dashboard") as demo:
    gr.Markdown("# 🌊 LLM Prompt Auto-Tuner System")
    gr.Markdown("Automated prompt optimization engine running via Hugging Face Inference API. This UI acts as a client connected safely to the background FastAPI endpoints.")
    
    with gr.Row():
        # LEFT COLUMN - CONFIGURATION
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configuration")
            
            model_id = gr.Textbox(
                label="Target Model ID", 
                value="meta-llama/Meta-Llama-3-8B-Instruct",
                info="The Hugging Face model to optimize for"
            )
            
            seed_prompt = gr.Textbox(
                label="Initial Seed Prompt", 
                lines=3,
                value="You are a helpful assistant. Summarize the text."
            )
            
            training_data = gr.Code(
                label="Training Data (JSON)",
                language="json",
                value='[\n  {\n    "input": "The Eiffel Tower is tall.",\n    "expected_keywords": ["Eiffel"]\n  }\n]'
            )
            
            optimize_btn = gr.Button("🚀 Run Optimization", variant="primary")
            
        # RIGHT COLUMN - RESULTS
        with gr.Column(scale=1):
            gr.Markdown("### 2. Live Results")
            
            best_prompt = gr.Textbox(label="Best Prompt Found", interactive=False)
            best_score = gr.Textbox(label="Best Score", interactive=False)
            
            log_output = gr.Code(label="Optimization Log Terminal", language="markdown", interactive=False, lines=10)

    # Wire up the button
    optimize_btn.click(
        fn=run_optimization,
        inputs=[model_id, seed_prompt, training_data],
        outputs=[best_prompt, best_score, log_output]
    )
