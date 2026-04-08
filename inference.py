import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from server.environment import TransplantEnv, TransplantGrader, TASKS
from models import TransplantAction, ActionType, TransportMode

# ── Required env variables ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN     = os.getenv("HF_TOKEN")     or os.getenv("API_KEY") or ""

TASK_NAME    = os.getenv("TASK_NAME", "task_easy_clear_match")
BENCHMARK    = "transplant-logistics-env"
MAX_STEPS    = 8
TEMPERATURE  = 0.0
MAX_TOKENS   = 300
SUCCESS_SCORE_THRESHOLD = 0.5

GRADER = TransplantGrader()

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

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert organ transplant coordinator.
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

def obs_to_prompt(obs) -> str:
    lines = [
        f"Step {obs.step} | Elapsed: {obs.elapsed_minutes:.0f} min",
        f"Task: {obs.task_description}",
    ]
    if obs.alerts:
        lines.append("\nALERTS:")
        for a in obs.alerts:
            lines.append(f"  {a}")
    lines.append("\nDONORS:")
    for d in obs.available_donors:
        remaining = d.viability_hours - d.cross_clamp_time_minutes / 60
        lines.append(
            f"  {d.donor_id}: {d.organ_type.value} blood={d.blood_type.value} "
            f"age={d.age} hosp={d.hospital_id} viability_left={remaining:.1f}h "
            f"KDPI={d.kdpi or 'N/A'} HLA={d.hla_antigens}"
        )
    lines.append("\nWAITLIST:")
    for r in obs.waitlist:
        lines.append(
            f"  {r.recipient_id}: needs={r.organ_needed.value} "
            f"blood={r.blood_type.value} age={r.age} "
            f"urgency={r.urgency.value} wait={r.wait_days}d "
            f"hosp={r.hospital_id} PRA={r.pra:.0%} "
            f"HLA_ab={r.hla_antibodies} MELD={r.meld_score or 'N/A'}"
        )
    if obs.pending_matches:
        lines.append("\nPENDING MATCHES:")
        for m in obs.pending_matches:
            lines.append(
                f"  {m.donor_id}->{m.recipient_id} "
                f"score={m.compatibility_score:.2f} "
                f"accepted={m.accepted} xm={m.crossmatch_pending}"
            )
    lines.append("\nYour action (JSON only):")
    return "\n".join(lines)

def parse_action(text: str) -> TransplantAction:
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
    data = json.loads(text[start:end])
    if data.get("transport_mode"):
        data["transport_mode"] = TransportMode(data["transport_mode"])
    return TransplantAction(**data)

def get_action(client: OpenAI, obs, history: List[dict]) -> tuple:
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
        fallback = TransplantAction(action_type=ActionType.PASS_TO_NEXT)
        return fallback, "pass_to_next", str(e)

# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(task_id: str, client: OpenAI, seed: int = 42) -> dict:
    env     = TransplantEnv(task_id)
    obs     = env.reset(seed=seed)
    history = []
    rewards = []
    steps   = 0
    score   = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            action, raw, error = get_action(client, obs, history)
            result = env.step(action)

            reward = result.reward or 0.0
            done   = result.done

            rewards.append(reward)
            steps = step

            log_step(
                step=step,
                action=action.action_type.value,
                reward=reward,
                done=done,
                error=error,
            )

            history.append({"role": "user",      "content": obs_to_prompt(obs)})
            history.append({"role": "assistant",  "content": raw})

            obs = result.observation
            if done:
                break

        # Grade the episode
        final_state = env.state()
        task        = TASKS[task_id]
        grades      = GRADER.grade(final_state, task)
        score       = grades["aggregate"]
        success     = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    log_end(success=success, steps=steps, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score,
            "success": success, "steps": steps, "rewards": rewards}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    client   = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    task_ids = list(TASKS.keys())
    results  = []

    for task_id in task_ids:
        r = run_episode(task_id, client, seed=42)
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