---
title: Prompt Optimizer
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# 🌊 LLM Prompt Auto-Tuner System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()

An automated prompt optimization engine designed to "auto-tune" LLM behavior for specific tasks. Built with a **Gymnasium-based Reinforcement Learning environment**, it systematically tests prompt modifiers to maximize model performance on target datasets.

## 🚀 Overview

The **LLM Prompt Auto-Tuner** automates the tedious process of manual prompt engineering. By treating prompt tuning as a sequence of decisions (actions) in an environment, the system identifies the most effective instruction variants for a given model (like Llama-3) and task (like JSON extraction or keyword-focused summarization).

## ✨ Key Features

- **🤖 Automated RL Optimization**: Uses a `Gymnasium` environment to treat prompt modification as a sequence of strategic actions.
- **📊 Gradio Dashboard**: A premium, interactive UI for configuring models, seed prompts, and training data with real-time logs.
- **🔌 FastAPI Backend**: Fully decoupled architecture with `/reset` and `/step` endpoints for programmatic control.
- **🤗 Hugging Face Integration**: Direct integration with the Hugging Face Inference API for low-latency model evaluation.
- **🏆 Custom Reward Model**: Advanced grading logic that evaluates LLM outputs based on keyword density, JSON validity, and constraint satisfaction.
- **🧬 OpenEnv Compliant**: Standardized observation and action spaces, ready for benchmarking and automated testing.

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/), [Gradio](https://gradio.app/)
- **RL Core**: [Gymnasium](https://gymnasium.farama.org/) (formerly OpenAI Gym)
- **Model Interface**: Hugging Face Inference API, OpenAI SDK
- **Language**: Python 3.9+
- **Infrastructure**: Docker-ready for deployment on Hugging Face Spaces.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/prompt-auto-tuner.git
   cd prompt-auto-tuner
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file or export your Hugging Face token:
   ```bash
   export HF_TOKEN="your_hugging_face_token_here"
   ```

## 🚀 Usage

### Running the Full App (UI + API)
```bash
python app.py
```
Access the dashboard at `http://localhost:7860`.

### API Integration
The system exposes two primary endpoints as per the OpenEnv standard:
- `POST /reset`: Initialize the environment with a model ID and seed prompt.
- `POST /step`: Execute an action (0-4) to apply a prompt modifier and get the reward/output.

## 🧠 Environment Design

### Action Space (Discrete 5)
The tuner explores five distinct "Prompt Engineering" strategies:
1. **Action 0**: Strict keyword focus with summary constraints.
2. **Action 1**: Word-count limiting and plain-text enforcement.
3. **Action 2**: Expert-persona alignment with factual term preservation.
4. **Action 3**: Structural formatting (Markdown headers/lists).
5. **Action 4**: Rewrite for maximum keyword precision.

### Reward Model
The environment calculates rewards based on:
- **Keyword Match**: Points for every expected keyword present in the output.
- **Constraint Satisfaction**: Penalties for excessive length or incorrect formatting.
- **Model Stability**: Rewards for consistent, high-quality responses.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---
Built with ❤️ for the LLM Community.
