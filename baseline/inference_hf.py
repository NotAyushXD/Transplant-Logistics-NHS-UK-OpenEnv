"""
Transplant Logistics — HuggingFace / Groq Baseline Inference
=============================================================
Three backend modes:
  1. "groq"  — Groq Inference API (FREE tier, fast, recommended)
               Requires: GROQ_API_KEY environment variable
  2. "api"   — HuggingFace Inference API (may require paid credits)
               Requires: HF_TOKEN environment variable
  3. "local" — Load model weights locally via `transformers`

Groq free models:
  llama-3.3-70b-versatile  ← default (12k TPM limit, best reasoning)
  llama-3.1-8b-instant     ← faster but 6k TPM limit (hits 413 on hard task)
  mixtral-8x7b-32768       ← long context

Usage:
  export GROQ_API_KEY=gsk_...
  python baseline/inference_hf.py --backend groq
  python baseline/inference_hf.py --backend groq --model llama-3.3-70b-versatile
  python baseline/inference_hf.py --backend local --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import TASKS, TransplantEnv, TransplantGrader
from models import ActionType, TransplantAction, TransportMode


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert organ transplant coordinator at NHS Blood and Transplant (NHSBT).

=== STRICT WORKFLOW — FOLLOW THIS ORDER ===
For EACH organ, complete ALL three steps before moving to the next organ:
  1. match_organ        — propose the donor → recipient pair
  2. dispatch_transport — send the organ IMMEDIATELY after matching
  3. After ALL organs are matched and dispatched → notify_team

NEVER skip dispatch_transport after a successful match.
NEVER attempt to match the next organ before dispatching the current one.

=== NHSBT RULES ===
1. ABO blood type is MANDATORY:
   O-  → any    |  O+  → O+,A+,B+,AB+  |  A-  → A-,A+,AB-,AB+
   A+  → A+,AB+ |  B-  → B-,B+,AB-,AB+ |  B+  → B+,AB+
   AB- → AB-,AB+|  AB+ → AB+ only
2. Organ type must EXACTLY match recipient need.
3. Cold ischaemia limits: heart 4h, lung 6h, liver 12h, kidney 24h.
4. PRA > 85% → request_crossmatch BEFORE match_organ.
   Positive crossmatch result → DO NOT match that pair. Move to next recipient.
5. Urgency: 1A > 1B > 2 > 7.
6. High-KDPI kidney (>0.70) → prefer older recipient (age >55).
7. Choose the FASTEST transport mode that fits the viability window.
   Use the REACHABLE flags in the observation — only match reachable recipients.

=== WHEN STUCK ===
If all valid actions are exhausted → notify_team.

Respond ONLY with a single valid JSON object. No markdown, no explanation.

{
  "action_type": "match_organ"|"dispatch_transport"|"request_crossmatch"|
                 "notify_team"|"decline_organ"|"reject_match"|"pass_to_next",
  "donor_id":       string or null,
  "recipient_id":   string or null,
  "transport_mode": "ground"|"charter"|"commercial"|null,
  "reason":         string or null
}"""


# ── Transport feasibility helper ───────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))

def _transport_min(lat1, lon1, lat2, lon2, mode):
    dist = _haversine_km(lat1, lon1, lat2, lon2)
    speed   = {"ground": 70.0,  "charter": 280.0,  "commercial": 500.0}[mode]
    overhead= {"ground": 15,    "charter": 60,      "commercial": 120}[mode]
    return (dist / speed) * 60 + overhead

def _best_transport(lat1, lon1, lat2, lon2, remaining_min):
    """
    Returns (best_mode, eta_min, reachable) where reachable=True if at
    least one transport mode gets the organ there before it expires.
    Tries fastest-first so we always show the best option.
    """
    for mode in ("charter", "commercial", "ground"):
        eta = _transport_min(lat1, lon1, lat2, lon2, mode)
        if eta < remaining_min:
            return mode, round(eta), True
    # Nothing works — return ground ETA anyway so the model sees why
    eta = _transport_min(lat1, lon1, lat2, lon2, "ground")
    return "ground", round(eta), False


# ── Observation → prompt ───────────────────────────────────────────────────────

BLOOD_COMPAT = {
    "O-":  ["O-","O+","A-","A+","B-","B+","AB-","AB+"],
    "O+":  ["O+","A+","B+","AB+"],
    "A-":  ["A-","A+","AB-","AB+"],
    "A+":  ["A+","AB+"],
    "B-":  ["B-","B+","AB-","AB+"],
    "B+":  ["B+","AB+"],
    "AB-": ["AB-","AB+"],
    "AB+": ["AB+"],
}

def obs_to_prompt(obs_dict: dict) -> str:
    donors   = obs_dict.get("available_donors", [])
    waitlist = obs_dict.get("waitlist", [])
    pending  = obs_dict.get("pending_matches", [])
    hospitals= {h["hospital_id"]: h for h in obs_dict.get("hospitals", [])}

    lines = [
        f"=== NHSBT Coordination | Step {obs_dict['step']} "
        f"| {obs_dict.get('elapsed_minutes', 0):.0f} min elapsed ===",
        f"Task: {obs_dict['task_description']}",
        "",
    ]

    # Urgent alerts first
    for a in obs_dict.get("alerts", []):
        lines.append(f"⚠ URGENT: {a}")
    if obs_dict.get("alerts"):
        lines.append("")

    # ── Workflow state banner ──────────────────────────────────────────────
    accepted       = [m for m in pending if m.get("accepted")]
    donor_ids_live = {d["donor_id"] for d in donors}
    needs_dispatch = [m for m in accepted if m["donor_id"] in donor_ids_live]

    lines.append("=== WORKFLOW STATE ===")
    if needs_dispatch:
        lines.append("▶ NEXT ACTION REQUIRED: dispatch_transport")
        for m in needs_dispatch:
            lines.append(f"   dispatch donor={m['donor_id']} (match accepted, not yet sent)")
    elif donors and waitlist:
        lines.append("▶ NEXT ACTION REQUIRED: match_organ (then immediately dispatch_transport)")
    elif donors:
        lines.append("▶ All recipients matched. Dispatch remaining organs, then notify_team.")
    else:
        lines.append("▶ All organs dispatched. Call notify_team.")
    lines.append("")

    # ── Donors with per-recipient feasibility ──────────────────────────────
    lines.append("=== AVAILABLE DONORS ===")
    for d in donors:
        elapsed_h   = d.get("cross_clamp_time_minutes", 0) / 60
        remaining_h = d.get("viability_hours", 0) - elapsed_h
        remaining_m = remaining_h * 60
        expiry_flag = " ⚠ EXPIRING VERY SOON" if remaining_h < 1.5 else (
                      " ⚠ EXPIRING SOON"      if remaining_h < 2.5 else "")

        dh = hospitals.get(d["hospital_id"], {})
        d_lat, d_lon = dh.get("lat", 0), dh.get("lon", 0)

        lines.append(
            f"\n  DONOR {d['donor_id']}"
            f" | organ={d['organ_type']} | blood={d['blood_type']} | age={d['age']}"
            f" | hospital={d['hospital_id']}"
            f" | viability_left={remaining_h:.1f}h ({remaining_m:.0f} min){expiry_flag}"
            f" | KDPI={d.get('kdpi') or 'N/A'}"
            f" | HLA_antigens={d.get('hla_antigens', [])}"
        )

        # Compatible recipients for this donor
        compat = [
            r for r in waitlist
            if r["organ_needed"] == d["organ_type"]
            and r["blood_type"] in BLOOD_COMPAT.get(d["blood_type"], [])
        ]

        if compat:
            lines.append("  Compatible recipients:")
            for r in compat:
                pra = r.get("pra", 0)

                # Transport feasibility
                rh = hospitals.get(r["hospital_id"], {})
                r_lat, r_lon = rh.get("lat", 0), rh.get("lon", 0)
                best_mode, eta_min, reachable = _best_transport(
                    d_lat, d_lon, r_lat, r_lon, remaining_m
                )
                reach_flag = (
                    f"✓ REACHABLE via {best_mode} ~{eta_min}min"
                    if reachable
                    else f"✗ NOT REACHABLE — fastest ({best_mode}) ~{eta_min}min > viability {remaining_m:.0f}min"
                )

                # PRA flag
                pra_flag = " ⚠ HIGH-PRA: crossmatch REQUIRED first" if pra > 0.85 else (
                           " [elevated PRA]" if pra > 0.50 else "")

                # Crossmatch status
                xm = next(
                    (m for m in pending
                     if m["donor_id"] == d["donor_id"]
                     and m["recipient_id"] == r["recipient_id"]
                     and m.get("crossmatch_pending")),
                    None
                )
                xm_flag = " [crossmatch already requested]" if xm else ""

                lines.append(
                    f"    → {r['recipient_id']}"
                    f" | urgency={r['urgency']} | wait={r['wait_days']}d"
                    f" | blood={r['blood_type']} | PRA={pra:.0%}{pra_flag}{xm_flag}"
                    f" | hospital={r['hospital_id']} | {reach_flag}"
                    f" | HLA_ab={r.get('hla_antibodies', [])}"
                    f" | MELD={r.get('meld_score') or 'N/A'}"
                    f" | eGFR={r.get('eGFR') or 'N/A'}"
                )
        else:
            lines.append("  ✗ No compatible recipients available for this donor.")

    # ── Blood-incompatible recipients (do not attempt) ─────────────────────
    incompat = [
        r for r in waitlist
        if not any(
            r["organ_needed"] == d["organ_type"]
            and r["blood_type"] in BLOOD_COMPAT.get(d["blood_type"], [])
            for d in donors
        )
    ]
    if incompat:
        lines.append("\n=== INCOMPATIBLE RECIPIENTS — DO NOT ATTEMPT ===")
        for r in incompat:
            lines.append(
                f"  {r['recipient_id']} | needs={r['organ_needed']}"
                f" | blood={r['blood_type']} | urgency={r['urgency']}"
                f" ← no compatible donor available"
            )

    lines.append("")
    lines.append("Your next action (JSON only):")
    return "\n".join(lines)


# ── Action parser ──────────────────────────────────────────────────────────────

def parse_action(raw: str) -> TransplantAction:
    raw = raw.strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found. Raw: {raw[:200]}")
    data = json.loads(raw[start:end])
    if data.get("transport_mode"):
        data["transport_mode"] = TransportMode(data["transport_mode"])
    return TransplantAction(**data)


# ── Backends ───────────────────────────────────────────────────────────────────

class GroqBackend:
    """
    Groq free inference API. Fast (~1s/step) with a generous free tier.
    Get a free key at: https://console.groq.com

    TPM limits (as of 2025):
      llama-3.3-70b-versatile : 12,000  ← recommended (avoids 413 on hard task)
      llama-3.1-8b-instant    :  6,000  ← too small for multi-step hard task history
    """
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, model_id: str, api_key: str):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("pip install groq")
        self.model_id = model_id or self.DEFAULT_MODEL
        self.client   = Groq(api_key=api_key)
        print(f"[Groq] Using model: {self.model_id}")

    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 300,
                 temperature: float = 0.1) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id, messages=messages,
            max_tokens=max_new_tokens, temperature=temperature,
        )
        return resp.choices[0].message.content


class HFApiBackend:
    def __init__(self, model_id: str, hf_token: str):
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("pip install huggingface_hub>=0.24.0")
        self.model_id = model_id
        self.client   = InferenceClient(model=model_id, token=hf_token)
        print(f"[HF API] Using model: {model_id}")

    def generate(self, messages, max_new_tokens=300, temperature=0.1):
        resp = self.client.chat_completion(
            messages=messages, max_tokens=max_new_tokens, temperature=temperature,
        )
        return resp.choices[0].message.content


class LocalModelBackend:
    def __init__(self, model_id: str):
        print(f"[Local] Loading model: {model_id}")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto", trust_remote_code=True,
        )
        self.model.eval()
        print(f"[Local] Loaded on {self.device} ✓")

    def generate(self, messages, max_new_tokens=300, temperature=0.1):
        import torch
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    task_id: str,
    backend,
    seed: int = 42,
    verbose: bool = True,
    nhs_csv_path: Optional[str] = None,
    max_history_turns: int = 4,       # FIX: keep only last N exchanges to avoid 413
) -> Dict[str, Any]:
    """
    max_history_turns: how many recent (user, assistant) pairs to keep in
    the message history. The current observation always captures full state,
    so dropping old turns doesn't lose critical information — it just reduces
    token count. Set to None to keep full history (risks 413 on long episodes).
    """
    env          = TransplantEnv(task_id, nhs_csv_path=nhs_csv_path)
    grader       = TransplantGrader()
    obs          = env.reset(seed=seed)
    history      = []      # list of {"role": ..., "content": ...}
    total_reward = 0.0
    step         = 0

    if verbose:
        print(f"\n{'═'*70}")
        print(f"Task:       {TASKS[task_id]['name']}")
        print(f"Difficulty: {TASKS[task_id]['difficulty']}")
        print(f"Max steps:  {TASKS[task_id]['max_steps']}")
        print(f"{'═'*70}")

    while True:
        user_msg = obs_to_prompt(obs.model_dump())
        history.append({"role": "user", "content": user_msg})

        # FIX: truncate history to last N (user, assistant) pairs
        # Each pair = 2 entries in the list, so keep last max_history_turns * 2
        if max_history_turns and len(history) > max_history_turns * 2:
            history = history[-(max_history_turns * 2):]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        t0 = time.time()
        try:
            raw = backend.generate(messages, max_new_tokens=300, temperature=0.1)
        except Exception as e:
            err_str = str(e)
            # FIX: auto-retry with aggressively truncated history on 413
            if "413" in err_str or "too large" in err_str.lower() or "rate_limit" in err_str.lower():
                if verbose:
                    print(f"  [413 context too large — retrying with last 2 turns]")
                short_history = history[-4:]   # last 2 pairs only
                messages_short = [{"role": "system", "content": SYSTEM_PROMPT}] + short_history
                try:
                    raw = backend.generate(messages_short, max_new_tokens=300, temperature=0.1)
                except Exception as e2:
                    if verbose:
                        print(f"  [retry also failed: {e2}]")
                    raw = '{"action_type": "pass_to_next"}'
            else:
                raise
        dt = time.time() - t0

        history.append({"role": "assistant", "content": raw})

        try:
            action = parse_action(raw)
        except Exception as e:
            if verbose:
                print(f"  [parse error @ step {step+1}] {e}")
                print(f"  Raw: {raw[:150]}")
            action = TransplantAction(action_type=ActionType.PASS_TO_NEXT)

        if verbose:
            print(f"\n[Step {step+1:02d}] {action.action_type.value:<22} "
                  f"donor={action.donor_id or '─':8} "
                  f"recipient={action.recipient_id or '─':6} "
                  f"mode={str(action.transport_mode or '─'):8} "
                  f"({dt:.1f}s)")

        result = env.step(action)
        total_reward += result.reward
        step += 1

        if verbose:
            info = result.info
            if info.get("error"):
                print(f"  ✗ ERROR:   {info['error']}")
            elif info.get("warning"):
                print(f"  ⚠ WARNING: {info['warning']}")
            elif info.get("crossmatch_result"):
                xm = info["crossmatch_result"]
                print(f"  ✓ Crossmatch: positive={xm['positive']} | {xm['recommendation']}")
            elif info.get("match"):
                m = info["match"]
                print(f"  ✓ Match: score={m['compatibility_score']:.2f} "
                      f"transport={m['transport_minutes']:.0f}min "
                      f"viability_left={m['remaining_viability_minutes']:.0f}min")
            elif info.get("transport_dispatched"):
                t = info["transport_dispatched"]
                print(f"  ✓ Dispatch: {t['from']} → {t['to']} "
                      f"via {t['mode']} ETA {t['eta_minutes']}min")
            elif info.get("notification_sent"):
                print(f"  ✓ Teams notified — {info['transplants_coordinated']} transplants")
            elif info.get("message"):
                print(f"  ℹ {info['message']}")
            for alert in info.get("alerts", []):
                print(f"  🚨 {alert}")
            print(f"     reward={result.reward:+.4f}  cumulative={total_reward:+.4f}")

        obs = result.observation
        if result.done:
            break

    final_state = env.state()
    grades      = grader.grade(final_state, TASKS[task_id])

    if verbose:
        print(f"\n── Episode complete {'─'*51}")
        print(f"   Steps taken:  {step}")
        print(f"   Total reward: {total_reward:+.4f}")
        print(f"\n── NHSBT Grade breakdown (0.0–1.0) {'─'*35}")
        for k, v in grades.items():
            bar = "█" * int(v * 24) if isinstance(v, float) else ""
            print(f"   {k:<25} {str(v):>6}  {bar}")

    return {
        "task_id":      task_id,
        "difficulty":   TASKS[task_id]["difficulty"],
        "steps":        step,
        "total_reward": round(total_reward, 4),
        **{f"grade_{k}": v for k, v in grades.items()},
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transplant Logistics NHS — baseline agent")
    parser.add_argument("--backend", default="groq", choices=["groq", "api", "local"])
    parser.add_argument(
        "--model", default=None,
        help=(
            "Model ID. Defaults: groq=llama-3.3-70b-versatile, "
            "api=Qwen/Qwen2.5-1.5B-Instruct, local=Qwen/Qwen2.5-1.5B-Instruct"
        ),
    )
    parser.add_argument("--task", default=None, choices=list(TASKS.keys()) + ["all"])
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--csv",   default=None)
    parser.add_argument(
        "--history-turns", type=int, default=4,
        help="Max (user, assistant) pairs to keep in context. "
             "Lower = fewer tokens = avoids 413. Default: 4.",
    )
    args = parser.parse_args()

    if args.backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            print("ERROR: set GROQ_API_KEY. Get a free key at https://console.groq.com")
            sys.exit(1)
        model   = args.model or GroqBackend.DEFAULT_MODEL
        backend = GroqBackend(model_id=model, api_key=api_key)

    elif args.backend == "api":
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("ERROR: set HF_TOKEN. Consider --backend groq (free).")
            sys.exit(1)
        model   = args.model or "Qwen/Qwen2.5-1.5B-Instruct"
        backend = HFApiBackend(model_id=model, hf_token=hf_token)

    else:
        model   = args.model or "Qwen/Qwen2.5-1.5B-Instruct"
        backend = LocalModelBackend(model_id=model)

    task_ids = (
        list(TASKS.keys())
        if (args.task is None or args.task == "all")
        else [args.task]
    )

    results = []
    for tid in task_ids:
        r = run_episode(
            tid, backend,
            seed               = args.seed,
            verbose            = not args.quiet,
            nhs_csv_path       = args.csv,
            max_history_turns  = args.history_turns,
        )
        results.append(r)

    print(f"\n{'═'*72}")
    print(f"NHSBT BASELINE SCORES  |  model={args.model or '(default)'}  backend={args.backend}")
    print(f"{'═'*72}")
    print(f"{'Task':<45} {'Diff':6} {'Aggregate':>9} {'Steps':>6}")
    print(f"{'─'*72}")
    for r in results:
        print(f"{r['task_id']:<45} {r['difficulty']:6} "
              f"{r['grade_aggregate']:>9.3f} {r['steps']:>6}")
    print(f"{'─'*72}")
    mean = sum(r["grade_aggregate"] for r in results) / len(results)
    print(f"{'MEAN AGGREGATE':>54} {mean:>9.3f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_scores_hf.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "backend": args.backend,
                   "seed": args.seed, "results": results}, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
