import gradio as gr
import requests
import json
import os

# Define the custom Theme
ocean_theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

# API Helper function
def run_optimization(model_id, seed_prompt, training_data):
    # This acts as an API Client hitting our own FastAPI endpoints
    # 1. We would typically hit /reset to start
    # 2. Then loop /step. For UI demonstration, we'll stream logs.
    
    yield "Starting Optimization Process...", "", "Initializing Environment..."
    
    try:
        # Ping the /reset endpoint
        requests.post("http://localhost:7860/reset")
        yield "Environment Reset Successful", "", "Running Step 1..."
        
        # Ping the /step endpoint
        # The true action logic requires RL integers. For UI Demo purposes, we simulate the log output.
        res = requests.post("http://localhost:7860/step", json={"action": 0})
        if res.status_code == 200:
            data = res.json()
            reward = data.get("reward", 0.0)
            yield "Testing Initial Prompt...", str(reward), f"Action 0 taken. Reward: {reward}"
        else:
            yield "Error", "Error", str(res.text)
            
    except Exception as e:
        yield "Connection Error", "Error", str(e)


with gr.Blocks(theme=ocean_theme, title="Prompt Auto-Tuner Dashboard") as demo:
    gr.Markdown("# 🌊 LLM Prompt Auto-Tuner System")
    gr.Markdown("Automated prompt optimization engine running via Hugging Face Inference API. This UI acts as a client connected safely to the background FastAPI endpoints.")
    
    with gr.Row():
        # LEFT COLUMN - CONFIGURATION
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configuration")
            
            model_id = gr.Textbox(
                label="Target Model ID", 
                value="meta-llama/Llama-3-8B-Instruct",
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
