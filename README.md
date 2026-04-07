---
title: Prompt Auto-Tuner
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

## ✨ Key Features

- **🤖 Automated RL Optimization**: Uses a `Gymnasium` environment to treat prompt modification as a sequence of strategic actions.
- **📊 Gradio Dashboard**: A premium, interactive UI for configuring models, seed prompts, and training data with real-time logs.
- **🔌 FastAPI Backend**: Fully decoupled architecture with `/reset` and `/step` endpoints for programmatic control.
- **🤗 Hugging Face Integration**: Direct integration with the Hugging Face Inference API for low-latency model evaluation.
- **🏆 Custom Reward Model**: Advanced grading logic that evaluates LLM outputs based on keyword density, JSON validity, and constraint satisfaction.
- **🧬 OpenEnv Compliant**: Standardized observation and action spaces, ready for benchmarking and automated testing.

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/), [Gradio](https://gradio.app/)
- **RL Core**: [Gymnasium](https://gymnasium.farama.org/)
- **Model Interface**: HF Inference API, OpenAI SDK
- **Language**: Python 3.9+

## 🚀 Usage

### Running Locally
```bash
python app.py
```
Access the dashboard at `http://localhost:7860`.

### API Integration
The system exposes two primary endpoints as per the OpenEnv standard:
- `POST /reset`: Initialize the environment with a model ID and seed prompt.
- `POST /step`: Execute an action (0-4) to apply a prompt modifier and get the reward/output.

## 📄 License
This project is licensed under the MIT License.
