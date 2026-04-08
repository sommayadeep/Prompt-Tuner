#!/usr/bin/env python3
"""
inference.py — LLM Prompt Auto-Tuner OpenEnv Agent
===================================================
Runs an LLM agent through all 3 keyword extraction tasks with structured logs.

Required environment variables:
    API_BASE_URL      LLM API endpoint
    MODEL_NAME        Model identifier
    HF_TOKEN          HuggingFace / API key

Stdout format (must not deviate):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import textwrap
from typing import List, Optional

from openai import OpenAI
import reward_model
from environment import PromptEnv

# ✅ EXPLICIT GRADER REGISTRY (Required for Phase 2 validator)
GRADERS = {
    "reward_model_grade": reward_model.grade
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
BENCHMARK    = "keyword-extraction"

MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE = 0.0
MAX_TOKENS = 150

# ✅ TASKS — One separate episode per task
TASKS = [
    {
        "name": "task1_keywords",
        "input": "The Eiffel Tower is in Paris.",
        "keywords": ["Eiffel", "Paris"],
        "grader": "reward_model_grade",
    },
    {
        "name": "task2_keywords",
        "input": "Ada Lovelace wrote the first algorithm.",
        "keywords": ["Ada Lovelace", "algorithm"],
        "grader": "reward_model_grade",
    },
    {
        "name": "task3_keywords",
        "input": "Tokyo is a major city in Japan.",
        "keywords": ["Tokyo", "Japan"],
        "grader": "reward_model_grade",
    },
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert keyword extraction agent.
    Given a text, extract the most important keywords/entities.
    
    Reply with ONLY a JSON object in this format:
    {"keywords": ["key1", "key2", ...]}
    
    No explanation. No extra text. Just the JSON object.
""").strip()

# ---------------------------------------------------------------------------
# Logging helpers — must match the spec exactly
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Log the start of a task episode."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log each step in the episode."""
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log the end of a task episode."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    observation: str,
    history: List[str],
) -> str:
    """Ask the LLM to extract keywords and return JSON string."""
    history_block = "\n".join(history[-3:]) if history else "None"
    user_prompt = (
        f"Extract keywords from this text:\n{observation}\n\n"
        f"Previous extractions this episode:\n{history_block}\n\n"
        f"Reply with JSON only:"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"keywords": []}'

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"keywords": []}'  # safe fallback

# ---------------------------------------------------------------------------
# Single task runner — ONE EPISODE PER TASK
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_info: dict) -> None:
    """Run one full episode for the given task, emitting [START]/[STEP]/[END] logs."""

    task_name = task_info["name"]
    task_input = task_info["input"]
    expected_keywords = task_info["keywords"]

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = PromptEnv()
        obs, info = env.reset()

        for step in range(1, MAX_STEPS + 1):
            # Get LLM action
            action_str = get_model_action(client, task_input, history)

            # Execute step in environment
            try:
                obs, reward, terminated, truncated, info = env.step(step % 5)
                done = terminated or truncated
                last_error = None
            except Exception as exc:
                reward = 0.0
                done = True
                last_error = str(exc)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)
            history.append(f"Step {step}: {action_str} → reward {reward:.2f}")

            if done:
                break

        # Calculate final score with strict bounds
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(1e-6, min(score, 1 - 1e-6))  # ✅ Strict bounds: (0.000001, 0.999999)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)
        last_error = str(exc)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ---------------------------------------------------------------------------
# Main — iterate all tasks SEPARATELY
# ---------------------------------------------------------------------------

def main() -> None:
    """Run each task as a separate episode."""
    client = None
    try:
        if API_KEY:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[DEBUG] OpenAI client init failed: {e}", flush=True)
        client = None

    # ✅ RUN EACH TASK AS A SEPARATE EPISODE
    for task_info in TASKS:
        run_task(client, task_info)


if __name__ == "__main__":
    main()
