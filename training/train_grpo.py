"""
Transplant Logistics — GRPO Training Loop
Connects the TransplantEnv to TRL's GRPOTrainer for post-training.

Architecture:
  TransplantEnv (HTTP) → rollout collector → GRPO reward signal → model update

Usage:
    # Local (env must be running on port 7860)
    python training/train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct

    # With custom env URL
    python training/train_grpo.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --env-url http://localhost:7860 \
        --steps 500 \
        --output ./checkpoints/transplant-grpo

Requirements:
    pip install trl transformers torch accelerate pydantic httpx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Lazy imports (heavy ML deps) ─────────────────────────────────────────────
def _import_ml():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    return torch, AutoTokenizer, AutoModelForCausalLM, GRPOConfig, GRPOTrainer

from server.environment import TASKS, TransplantEnv, TransplantGrader
from models import (
    ActionType, TransplantAction, TransplantObservation, TransportMode,
)

GRADER = TransplantGrader()

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert organ transplant coordinator at a national allocation centre.
Allocate donor organs to recipients following UNOS/NOTTO protocols.

Rules (non-negotiable):
1. Blood type compatibility is mandatory — O- is universal donor.
2. Organ type must match recipient need exactly.
3. Cold ischaemia limits: heart 4h, lung 6h, liver 12h, kidney 24h.
4. Recipients with PRA > 0.50 MUST have a crossmatch before matching.
5. Status 1A > 1B > 2 for urgency priority.
6. High-KDPI kidneys (>0.70) suit older recipients (age >55).
7. Always end with notify_team after allocations are complete.

Respond ONLY with a valid JSON action. No explanation outside JSON.

Action schema:
{
  "action_type": "match_organ" | "dispatch_transport" | "request_crossmatch" |
                 "notify_team" | "decline_organ" | "reject_match" | "pass_to_next",
  "donor_id":       string | null,
  "recipient_id":   string | null,
  "transport_mode": "ground" | "charter" | "commercial" | null,
  "reason":         string | null
}"""


def obs_to_text(obs: TransplantObservation) -> str:
    """Render observation as a compact text prompt."""
    lines = [
        f"=== Transplant Coordination | Step {obs.step} | {obs.elapsed_minutes:.0f} min elapsed ===",
        f"Task: {obs.task_description}",
    ]

    if obs.alerts:
        lines.append("\n⚠ URGENT ALERTS:")
        for a in obs.alerts:
            lines.append(f"  {a}")

    lines.append("\nAVAILABLE DONORS:")
    for d in obs.available_donors:
        elapsed_h = d.cross_clamp_time_minutes / 60
        remaining_h = d.viability_hours - elapsed_h
        lines.append(
            f"  {d.donor_id}: {d.organ_type.value} | blood={d.blood_type.value} "
            f"| age={d.age} | hosp={d.hospital_id} "
            f"| viability_left={remaining_h:.1f}h "
            f"| KDPI={d.kdpi or 'N/A'} | HLA={d.hla_antigens}"
        )

    lines.append("\nWAITLIST:")
    for r in obs.waitlist:
        lines.append(
            f"  {r.recipient_id}: needs={r.organ_needed.value} | blood={r.blood_type.value} "
            f"| age={r.age} | urgency={r.urgency.value} | wait={r.wait_days}d "
            f"| hosp={r.hospital_id} | PRA={r.pra:.0%} "
            f"| MELD={r.meld_score or 'N/A'} | HLA_ab={r.hla_antibodies}"
        )

    if obs.pending_matches:
        lines.append("\nPENDING MATCHES:")
        for m in obs.pending_matches:
            lines.append(
                f"  {m.donor_id}→{m.recipient_id} score={m.compatibility_score:.2f} "
                f"accepted={m.accepted} xm={m.crossmatch_pending}"
            )

    lines.append("\nYour action (JSON only):")
    return "\n".join(lines)


def parse_action_from_response(text: str) -> TransplantAction:
    """Extract and parse a JSON action from model output."""
    text = text.strip()
    # Strip markdown code fences if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            if "{" in part:
                text = part.strip()
                if text.startswith("json"):
                    text = text[4:].strip()
                break
    # Find first { ... } block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    data = json.loads(text[start:end])
    if "transport_mode" in data and data["transport_mode"]:
        data["transport_mode"] = TransportMode(data["transport_mode"])
    return TransplantAction(**data)


# ── Rollout collection ────────────────────────────────────────────────────────

def collect_rollout(
    task_id: str,
    model,
    tokenizer,
    device: str,
    seed: int = 42,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    backend: str = "hf",
    groq_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run one full episode and return the trajectory as
    {prompts, responses, rewards} for GRPO.
    """
    env  = TransplantEnv(task_id)
    obs  = env.reset(seed=seed)
    task = TASKS[task_id]

    prompts:   List[str] = []
    responses: List[str] = []
    rewards:   List[float] = []

    for _ in range(task["max_steps"]):
        user_text = obs_to_text(obs)
        # Build chat-formatted prompt
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_text},
        ]
        if backend == "hf":
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

            import torch
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # Decode only the newly generated tokens
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        else:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content
            prompt_text = json.dumps(messages)

        # Parse action
        try:
            action = parse_action_from_response(response_text)
        except Exception:
            action = TransplantAction(action_type=ActionType.PASS_TO_NEXT)
            response_text = '{"action_type": "pass_to_next"}'

        result = env.step(action)

        prompts.append(prompt_text)
        responses.append(response_text)
        rewards.append(result.reward)

        obs = result.observation
        if result.done:
            break

    # Terminal grader score as bonus on final step
    final_state = env.state()
    grade = GRADER.grade(final_state, task)
    if rewards:
        rewards[-1] += grade["aggregate"] * 0.5   # terminal bonus

    return {
        "task_id":    task_id,
        "prompts":    prompts,
        "responses":  responses,
        "rewards":    rewards,
        "grade":      grade,
        "steps":      len(prompts),
    }


# ── GRPO dataset builder ──────────────────────────────────────────────────────

def build_grpo_dataset(
    task_ids: List[str],
    model,
    tokenizer,
    device: str,
    n_rollouts_per_task: int = 4,
    seed_start: int = 0,
    backend: str = "hf",
    groq_client: Optional[Any] = None,
) -> "datasets.Dataset":
    """
    Collect rollouts across all tasks and format them as a
    HuggingFace Dataset for GRPOTrainer.
    """
    from datasets import Dataset

    all_prompts:   List[str]   = []
    all_responses: List[str]   = []
    all_rewards:   List[float] = []

    for task_id in task_ids:
        for i in range(n_rollouts_per_task):
            seed = seed_start + i
            rollout = collect_rollout(
                task_id, model, tokenizer, device, seed=seed,
                backend=backend, groq_client=groq_client
            )
            all_prompts.extend(rollout["prompts"])
            all_responses.extend(rollout["responses"])
            all_rewards.extend(rollout["rewards"])
            print(
                f"  [{task_id}] rollout {i+1}/{n_rollouts_per_task} "
                f"steps={rollout['steps']} "
                f"grade={rollout['grade']['aggregate']:.3f}"
            )

    return Dataset.from_dict({
        "prompt":   all_prompts,
        "response": all_responses,
        "reward":   all_rewards,
    })


# ── Reward function for GRPOTrainer ──────────────────────────────────────────

def transplant_reward_fn(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Stateless reward function called by GRPOTrainer.
    Parses the completion as a TransplantAction and scores it
    against heuristic rules (since we can't run the env here).
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            action = parse_action_from_response(completion)
            r = _score_action_heuristic(action, prompt)
        except Exception:
            r = -0.2   # malformed JSON penalty
        rewards.append(r)
    return rewards


def _score_action_heuristic(action: TransplantAction, prompt: str) -> float:
    """
    Fast heuristic scorer — used when env state isn't available.
    Rewards well-formed, protocol-compliant actions.
    """
    r = 0.0

    # Valid action type
    if action.action_type in (ActionType.MATCH_ORGAN, ActionType.DISPATCH_TRANSPORT,
                               ActionType.REQUEST_CROSSMATCH):
        r += 0.1

    # Has required fields for match
    if action.action_type == ActionType.MATCH_ORGAN:
        if action.donor_id and action.recipient_id:
            r += 0.2
        else:
            r -= 0.2

    # Transport mode specified for dispatch
    if action.action_type == ActionType.DISPATCH_TRANSPORT:
        if action.transport_mode:
            r += 0.1
        if not action.donor_id:
            r -= 0.1

    # Crossmatch has both IDs
    if action.action_type == ActionType.REQUEST_CROSSMATCH:
        if action.donor_id and action.recipient_id:
            r += 0.15

    # Notify team — valid terminal action
    if action.action_type == ActionType.NOTIFY_TEAM:
        r += 0.05

    # Decline with reason is better than blind decline
    if action.action_type == ActionType.DECLINE_ORGAN and action.reason:
        r += 0.05

    return round(r, 4)


# ── Main training script ──────────────────────────────────────────────────────

def train(args):
    # Auto-detect Groq backend if not explicitly set
    if args.backend == "hf" and "llama" in args.model.lower() and "/" not in args.model:
        args.backend = "groq"

    device = "cpu"
    tokenizer = None
    model = None
    groq_client = None

    print(f"Backend: {args.backend}")
    print(f"Model:   {args.model}")
    print(f"Tasks:   {list(TASKS.keys())}")

    if args.backend == "hf":
        torch, AutoTokenizer, AutoModelForCausalLM, GRPOConfig, GRPOTrainer = _import_ml()
        from datasets import Dataset
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device:  {device}")

        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)
    else:
        import groq
        if not os.environ.get("GROQ_API_KEY"):
            print("ERROR: GROQ_API_KEY environment variable is required when using the groq backend.")
            sys.exit(1)
        groq_client = groq.Groq()
        model = args.model

    # ── Phase 1: collect warm-start rollouts ─────────────────────────────
    print("\nPhase 1: Collecting warm-start rollouts...")
    task_ids = list(TASKS.keys())
    
    # Optional datasets import for Dataset.from_dict
    from datasets import Dataset
    
    dataset  = build_grpo_dataset(
        task_ids, model, tokenizer, device,
        n_rollouts_per_task=args.rollouts_per_task,
        backend=args.backend,
        groq_client=groq_client
    )
    print(f"Dataset: {len(dataset)} samples")

    # ── Phase 2: GRPO training ────────────────────────────────────────────
    if args.backend == "hf":
        print("\nPhase 2: GRPO training...")
        config = GRPOConfig(
            output_dir                = args.output,
            num_train_epochs          = args.epochs,
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_accum,
            learning_rate             = args.lr,
            max_new_tokens            = 200,
            temperature               = 0.7,
            num_generations           = args.num_generations,
            beta                      = 0.04,           # KL penalty coefficient
            logging_steps             = 10,
            save_steps                = 100,
            report_to                 = "none",
            # Task-specific
            max_prompt_length         = 1024,
            max_completion_length     = 256,
        )

        trainer = GRPOTrainer(
            model         = model,
            args          = config,
            reward_funcs  = transplant_reward_fn,
            train_dataset = dataset,
            processing_class = tokenizer,
        )

        print("Starting GRPO training...")
        trainer.train()
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"\nModel saved to {args.output}")
    else:
        print("\nPhase 2: Skipping GRPO training (not supported for API models).")
        os.makedirs(args.output, exist_ok=True)

    # ── Phase 3: Evaluate trained model ──────────────────────────────────
    print("\nPhase 3: Post-training evaluation...")
    scores = {}
    for task_id in task_ids:
        rollout = collect_rollout(
            task_id, model, tokenizer, device, seed=999,
            backend=args.backend, groq_client=groq_client
        )
        scores[task_id] = rollout["grade"]["aggregate"]
        print(f"  {task_id}: {scores[task_id]:.3f}")

    mean = sum(scores.values()) / len(scores)
    print(f"\n  Mean aggregate score: {mean:.3f}")

    results = {
        "model":           args.model,
        "output":          args.output,
        "post_train_scores": scores,
        "mean_aggregate":  mean,
    }
    out_path = os.path.join(args.output, "eval_results.json")
    os.makedirs(args.output, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GRPO post-training for Transplant Logistics OpenEnv"
    )
    parser.add_argument("--model",    default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model (HF hub path)")
    parser.add_argument("--output",   default="./checkpoints/transplant-grpo",
                        help="Output directory for trained model")
    parser.add_argument("--env-url",  default="http://localhost:7860",
                        help="TransplantEnv HTTP server URL")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch-size",   type=int,   default=2)
    parser.add_argument("--grad-accum",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="GRPO generation group size")
    parser.add_argument("--rollouts-per-task", type=int, default=4,
                        help="Warm-start rollouts per task")
    parser.add_argument("--backend", default="hf", choices=["hf", "groq"],
                        help="Inference backend (always hf for actual training)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
