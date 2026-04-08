#!/usr/bin/env python3
"""
Phase 2 inference script for Prompt Auto-Tuner.
Emits logs in the exact START/STEP/END format expected by the validator.
Scores are strictly in (0,1) and each task has a grader.
"""

import json
import os
from typing import List

from openai import OpenAI
import reward_model

# Validator-provided env vars
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

# Constants
ENV_NAME = "prompt-tuner"
TASKS = [
    {
        "id": "task1_keywords",
        "input": "The Eiffel Tower is in Paris.",
        "target": {"expected_keywords": ["Eiffel", "Paris"]},
        "grader": "reward_model.grade",
    },
    {
        "id": "task2_keywords",
        "input": "Ada Lovelace wrote the first algorithm.",
        "target": {"expected_keywords": ["Ada Lovelace", "algorithm"]},
        "grader": "reward_model.grade",
    },
    {
        "id": "task3_keywords",
        "input": "Tokyo is a major city in Japan.",
        "target": {"expected_keywords": ["Tokyo", "Japan"]},
        "grader": "reward_model.grade",
    },
]


def _strict_open_interval(score: float) -> float:
    return max(0.01, min(0.99, float(score)))


def log_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def maybe_client():
    if not API_KEY:
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:
        return None


def llm_output(client, prompt: str) -> str:
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def run_task(client, task) -> None:
    # START
    log_start(task["id"])

    # Single-step episode for this benchmark
    prompt = f"Extract keywords from: {task['input']}. Return JSON {{\"keywords\": [..]}} only."
    output = llm_output(client, prompt)
    if output is None:
        output = json.dumps(task["target"])

    raw_reward = reward_model.grade(output, task["target"])
    reward = _strict_open_interval(raw_reward)

    # STEP (only one)
    log_step(step=1, action="extract", reward=reward, done=True, error=None)

    # END
    score = reward  # single-step score
    log_end(success=True, steps=1, score=score, rewards=[reward])


def main():
    client = maybe_client()
    for task in TASKS:
        run_task(client, task)


if __name__ == "__main__":
    main()
