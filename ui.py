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
    yield "Starting Optimization Process...", "", "Initializing Environment..."
    
    try:
        td = json.loads(training_data)
        
        # Ping the /reset endpoint with UI inputs
        requests.post("http://127.0.0.1:7860/reset", json={"seed_prompt": seed_prompt, "training_data": td, "model_id": model_id})
        yield "Environment Reset Successful", "", "Running Steps...\n"
        
        logs = "Environment initialised with UI inputs.\n"
        best_score = -1.0
        best_prompt = seed_prompt
        
        # Mocking an RL loop of 5 actions to match the environment modifiers
        for action in range(5):
            yield f"Testing Action {action}...", str(best_score if best_score >= 0 else 0), logs + f"Running action {action}...\n"
            
            res = requests.post("http://127.0.0.1:7860/step", json={"action": action})
            if res.status_code == 200:
                data = res.json()
                reward = data.get("reward", 0.0)
                info = data.get("info", {})
                output_str = info.get("output", "")
                prompt_str = info.get("prompt", seed_prompt)
                
                logs += f"> Action {action} taken. Reward: {reward} | Output Preview: {output_str[:40]}...\n"
                
                if reward > best_score:
                    best_score = reward
                    best_prompt = prompt_str
            else:
                logs += f"Error: {res.text}\n"
                break
                
        yield best_prompt, str(best_score), logs + "\nOptimization Complete!"
            
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
