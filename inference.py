"""
Transplant Logistics — Inference Script (OpenEnv Round 1)
Runs an LLM agent against the deployed HF Space via HTTP API.

Required environment variables:
    API_BASE_URL   — LLM endpoint
    MODEL_NAME     — model identifier
    HF_TOKEN       — API key

Optional:
    ENV_URL        — base URL of the deployed environment
                     (default: https://notayushxd-transplant-logistics-env.hf.space)
    TASK_NAME      — single task to run (default: runs all tasks)
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── Required env variables ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN     = os.getenv("HF_TOKEN")     or os.getenv("API_KEY")

# Environment URL — the deployed HF Space
ENV_URL      = os.getenv("ENV_URL", "https://notayushxd-transplant-logistics-env.hf.space")

TASK_NAME    = os.getenv("TASK_NAME", "")
BENCHMARK    = "transplant-logistics-env"
TEMPERATURE  = 0.0
MAX_TOKENS   = 300
SUCCESS_SCORE_THRESHOLD = 0.5

HTTP_TIMEOUT = 30.0


# ── Log format — must match exactly ──────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int,
            score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── HTTP helpers ──────────────────────────────────────────────────────────────
_http = httpx.Client(base_url=ENV_URL, timeout=HTTP_TIMEOUT)


def env_reset(task_id: str, seed: int = 42) -> Dict[str, Any]:
    """POST /reset -> observation dict."""
    resp = _http.post("/reset", json={"task_id": task_id, "seed": seed})
    resp.raise_for_status()
    return resp.json()


def env_step(task_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    """POST /step -> {observation, reward, done, info}."""
    resp = _http.post("/step", json={"task_id": task_id, "action": action})
    resp.raise_for_status()
    return resp.json()


def env_state(task_id: str) -> Dict[str, Any]:
    """GET /state -> full internal state."""
    resp = _http.get("/state", params={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def env_grade(task_id: str) -> Dict[str, Any]:
    """POST /grade -> grading scores."""
    resp = _http.post("/grade", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def env_tasks() -> Dict[str, Any]:
    """GET /tasks -> task metadata."""
    resp = _http.get("/tasks")
    resp.raise_for_status()
    return resp.json()


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert organ transplant coordinator at NHSBT (NHS Blood and Transplant).
Allocate donor organs to recipients following these rules:
1. Blood type must be compatible — O- is universal donor.
2. Organ type must match what the recipient needs.
3. Hearts expire in 4h, lungs 6h, livers 12h, kidneys 24h.
4. Recipients with PRA > 0.50 need a crossmatch first.
5. Prioritise Status 1A > 1B > 2.
6. End with notify_team when done.

Reply ONLY with valid JSON — no explanation outside the JSON.

Action schema:
{
  "action_type": "match_organ" | "dispatch_transport" | "request_crossmatch" |
                 "notify_team" | "decline_organ" | "pass_to_next",
  "donor_id":       string | null,
  "recipient_id":   string | null,
  "transport_mode": "ground" | "charter" | "commercial" | null,
  "reason":         string | null
}"""


def obs_to_prompt(obs: Dict[str, Any]) -> str:
    """Convert observation dict (from HTTP JSON) to a concise text prompt."""
    donors  = obs.get("available_donors", [])
    recips  = obs.get("waitlist", [])
    alerts  = obs.get("alerts", [])
    pending = obs.get("pending_matches", [])
    elapsed = obs.get("elapsed_minutes", 0)

    lines = [
        f"Step {obs['step']} | Elapsed: {elapsed:.0f} min",
        f"Task: {obs.get('task_description', '')}",
    ]
    if alerts:
        lines.append("\nALERTS:")
        for a in alerts:
            lines.append(f"  {a}")

    lines.append("\nDONORS:")
    for d in donors:
        remaining = d.get("viability_hours", 0) - d.get("cross_clamp_time_minutes", 0) / 60
        lines.append(
            f"  {d['donor_id']}: {d['organ_type']} blood={d['blood_type']} "
            f"age={d['age']} hosp={d['hospital_id']} viability_left={remaining:.1f}h "
            f"KDPI={d.get('kdpi', 'N/A') or 'N/A'} HLA={d.get('hla_antigens', [])}"
        )

    lines.append("\nWAITLIST:")
    for r in recips:
        pra = r.get("pra", 0)
        lines.append(
            f"  {r['recipient_id']}: needs={r['organ_needed']} "
            f"blood={r['blood_type']} age={r['age']} "
            f"urgency={r['urgency']} wait={r.get('wait_days', 0)}d "
            f"hosp={r['hospital_id']} PRA={pra:.0%} "
            f"HLA_ab={r.get('hla_antibodies', [])} MELD={r.get('meld_score') or 'N/A'}"
        )

    if pending:
        lines.append("\nPENDING MATCHES:")
        for m in pending:
            lines.append(
                f"  {m['donor_id']}->{m['recipient_id']} "
                f"score={m['compatibility_score']:.2f} "
                f"accepted={m.get('accepted', False)} xm={m.get('crossmatch_pending', False)}"
            )

    lines.append("\nYour action (JSON only):")
    return "\n".join(lines)


def parse_action(text: str) -> Dict[str, Any]:
    """Parse LLM JSON output into an action dict for the HTTP API."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            if "{" in part:
                text = part.strip()
                if text.startswith("json"):
                    text = text[4:].strip()
                break
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON found")
    return json.loads(text[start:end])


def get_action(client: OpenAI, obs: Dict, history: List[dict]) -> tuple:
    user_msg = obs_to_prompt(obs)
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_msg}]
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        action = parse_action(raw)
        return action, raw, None
    except Exception as e:
        fallback = {"action_type": "pass_to_next"}
        return fallback, "pass_to_next", str(e)


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(task_id: str, task_meta: Dict, client: OpenAI,
                seed: int = 42) -> dict:
    max_steps = task_meta.get("max_steps", 20)

    obs     = env_reset(task_id, seed=seed)
    history: List[dict] = []
    rewards: List[float] = []
    steps   = 0
    score   = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, max_steps + 1):
            action, raw, error = get_action(client, obs, history)
            result = env_step(task_id, action)

            reward = result.get("reward", 0.0) or 0.0
            done   = result.get("done", False)

            rewards.append(reward)
            steps = step

            action_name = action.get("action_type", "unknown")
            log_step(
                step=step,
                action=action_name,
                reward=reward,
                done=done,
                error=error,
            )

            # Keep conversation history for multi-turn context
            history.append({"role": "user",      "content": obs_to_prompt(obs)})
            history.append({"role": "assistant",  "content": raw})

            obs = result.get("observation", {})

            if done:
                break

        # Grade via HTTP
        grades = env_grade(task_id)
        scores = grades.get("scores", grades)
        score  = scores.get("aggregate", 0.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    log_end(success=success, steps=steps, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score,
            "success": success, "steps": steps, "rewards": rewards}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    # Fetch available tasks from the running environment
    all_tasks = env_tasks()
    print(f"[DEBUG] Connected to {ENV_URL}", flush=True)
    print(f"[DEBUG] Available tasks: {list(all_tasks.keys())}", flush=True)

    # Run specified task or all tasks
    if TASK_NAME and TASK_NAME in all_tasks:
        task_ids = [TASK_NAME]
    else:
        task_ids = list(all_tasks.keys())

    results = []
    for task_id in task_ids:
        r = run_episode(task_id, all_tasks[task_id], client, seed=42)
        results.append(r)

    # Summary
    print("\n" + "="*60, flush=True)
    print("BASELINE SCORES", flush=True)
    print("="*60, flush=True)
    for r in results:
        print(
            f"{r['task_id']:<45} score={r['score']:.3f} "
            f"success={r['success']}",
            flush=True,
        )
    mean = sum(r["score"] for r in results) / len(results)
    print(f"\nMean aggregate score: {mean:.3f}", flush=True)


if __name__ == "__main__":
    main()
