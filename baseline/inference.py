"""
Transplant Logistics — Baseline Inference Script
Runs an LLM agent (via OpenAI-compatible API) against all 3 tasks
and produces reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=sk-...
    export OPENAI_BASE_URL=https://api.openai.com/v1   # or any compatible endpoint
    export MODEL_NAME=gpt-4o-mini                       # default

    python baseline/inference.py
    python baseline/inference.py --task task_easy_clear_match --seed 42
"""

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from server.environment import TASKS, TransplantEnv, TransplantGrader
from models import ActionType, TransplantAction, TransportMode

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY  = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL    = os.environ.get("MODEL_NAME", "gpt-4o-mini")

SYSTEM_PROMPT = """\
You are an expert organ transplant coordinator at a national allocation centre.
Your job is to allocate available donor organs to the most appropriate recipients
on the waiting list, following UNOS/NOTTO protocols.

Key rules:
1. Blood type compatibility is mandatory — check it first.
2. Organ type must match what the recipient needs.
3. Cold ischaemia time is critical — match hearts within 4 h, lungs 6 h,
   livers 12 h, kidneys 24 h of procurement.
4. High-PRA recipients (PRA > 0.50) REQUIRE a crossmatch before matching.
5. Prioritise by urgency: Status 1A > 1B > 2.
6. For kidneys, prefer donors with similar age and low KDPI to younger recipients;
   high KDPI donors are acceptable for older recipients.
7. Always notify the surgical team after completing allocations.

Respond ONLY with a valid JSON object matching the action schema.
Do not add explanation outside the JSON.

Action schema:
{
  "action_type": one of [
    "match_organ", "reject_match", "dispatch_transport",
    "request_crossmatch", "notify_team", "decline_organ", "pass_to_next"
  ],
  "donor_id":       string or null,
  "recipient_id":   string or null,
  "transport_mode": "ground" | "charter" | "commercial" | null,
  "message":        string or null,
  "reason":         string or null
}
"""


def obs_to_prompt(obs_dict: dict) -> str:
    """Convert observation dict to a concise text prompt."""
    donors = obs_dict.get("available_donors", [])
    recips = obs_dict.get("waitlist", [])
    alerts = obs_dict.get("alerts", [])
    pending = obs_dict.get("pending_matches", [])
    elapsed = obs_dict.get("elapsed_minutes", 0)

    lines = [
        f"Step {obs_dict['step']} | Elapsed: {elapsed:.0f} min",
        f"Task: {obs_dict['task_description']}",
        "",
    ]

    if alerts:
        lines.append("⚠ ALERTS:")
        for a in alerts:
            lines.append(f"  {a}")
        lines.append("")

    lines.append("AVAILABLE DONORS:")
    for d in donors:
        elapsed_h = d.get("cross_clamp_time_minutes", 0) / 60
        remaining_h = d.get("viability_hours", 0) - elapsed_h
        lines.append(
            f"  {d['donor_id']} | {d['organ_type']} | blood={d['blood_type']} "
            f"| age={d['age']} | hospital={d['hospital_id']} "
            f"| KDPI={d.get('kdpi','N/A')} "
            f"| viability remaining={remaining_h:.1f}h "
            f"| HLA={d.get('hla_antigens', [])}"
        )

    lines.append("")
    lines.append("WAITLIST RECIPIENTS:")
    for r in recips:
        lines.append(
            f"  {r['recipient_id']} | needs={r['organ_needed']} "
            f"| blood={r['blood_type']} | age={r['age']} "
            f"| urgency={r['urgency']} | wait={r['wait_days']}d "
            f"| hospital={r['hospital_id']} | PRA={r['pra']:.0%} "
            f"| HLA_antibodies={r.get('hla_antibodies', [])} "
            f"| MELD={r.get('meld_score','N/A')}"
        )

    if pending:
        lines.append("")
        lines.append("PENDING MATCHES:")
        for m in pending:
            lines.append(
                f"  {m['donor_id']} → {m['recipient_id']} "
                f"| score={m['compatibility_score']:.2f} "
                f"| xm_pending={m.get('crossmatch_pending', False)} "
                f"| accepted={m.get('accepted', False)}"
            )

    lines.append("")
    lines.append("Decide your next action. Reply ONLY with valid JSON.")
    return "\n".join(lines)


def parse_action(raw: str) -> TransplantAction:
    """Parse LLM JSON output into a TransplantAction."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())
    # Map transport_mode string
    tm = data.get("transport_mode")
    if tm:
        data["transport_mode"] = TransportMode(tm)
    return TransplantAction(**data)


def run_episode(task_id: str, client: OpenAI, seed: int = 42,
                verbose: bool = True) -> dict:
    env     = TransplantEnv(task_id)
    grader  = TransplantGrader()
    obs     = env.reset(seed=seed)
    history = []   # conversation history for multi-turn context
    total_reward = 0.0
    step = 0

    if verbose:
        print(f"\n{'═'*60}")
        print(f"Task: {TASKS[task_id]['name']}  [{TASKS[task_id]['difficulty']}]")
        print(f"{'═'*60}")

    while True:
        obs_dict = obs.model_dump()
        user_msg = obs_to_prompt(obs_dict)
        history.append({"role": "user", "content": user_msg})

        # Call the model
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            temperature=0.0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content
        history.append({"role": "assistant", "content": raw})

        # Parse action
        try:
            action = parse_action(raw)
        except Exception as e:
            if verbose:
                print(f"  [parse error @ step {step}]: {e}\n  raw: {raw[:120]}")
            action = TransplantAction(action_type=ActionType.PASS_TO_NEXT)

        if verbose:
            print(f"\n[Step {step+1}] {action.action_type.value} "
                  f"donor={action.donor_id} recipient={action.recipient_id} "
                  f"mode={action.transport_mode}")

        result = env.step(action)
        total_reward += result.reward
        step += 1

        if verbose:
            info = result.info
            if info.get("error"):
                print(f"  ✗ {info['error']}")
            elif info.get("warning"):
                print(f"  ⚠ {info['warning']}")
            elif info.get("match"):
                m = info["match"]
                print(f"  ✓ Match: score={m['compatibility_score']:.2f} "
                      f"transport={m['transport_minutes']:.0f}min")
            elif info.get("transport_dispatched"):
                t = info["transport_dispatched"]
                print(f"  ✓ Transport: {t['from']} → {t['to']} "
                      f"via {t['mode']} ETA {t['eta_minutes']}min")
            elif info.get("crossmatch_result"):
                xm = info["crossmatch_result"]
                print(f"  ✓ Crossmatch: positive={xm['positive']} | {xm['recommendation']}")
            elif info.get("notification_sent"):
                print(f"  ✓ Teams notified — {info['transplants_coordinated']} transplants coordinated")
            for alert in info.get("alerts", []):
                print(f"  🚨 {alert}")
            print(f"  reward={result.reward:+.4f}  cumulative={total_reward:+.4f}")

        obs = result.observation

        if result.done:
            break

    final_state = env.state()
    task        = TASKS[task_id]
    grades      = grader.grade(final_state, task)

    if verbose:
        print(f"\n── Final grades ──────────────────────────────")
        for k, v in grades.items():
            bar = "█" * int(v * 20) if isinstance(v, float) else ""
            print(f"  {k:25s} {v!s:6} {bar}")
        print(f"  total_reward:             {total_reward:+.4f}")

    return {
        "task_id":      task_id,
        "difficulty":   TASKS[task_id]["difficulty"],
        "steps":        step,
        "total_reward": round(total_reward, 4),
        **{f"grade_{k}": v for k, v in grades.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Transplant Logistics baseline inference"
    )
    parser.add_argument("--task", default=None,
                        choices=list(TASKS.keys()) + ["all"],
                        help="Task to run (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: set OPENAI_API_KEY environment variable")
        sys.exit(1)

    client   = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    task_ids = list(TASKS.keys()) if (args.task is None or args.task == "all") \
               else [args.task]

    results = []
    for tid in task_ids:
        r = run_episode(tid, client, seed=args.seed,
                        verbose=not args.quiet)
        results.append(r)

    # Summary table
    print(f"\n{'═'*70}")
    print("BASELINE SCORES")
    print(f"{'═'*70}")
    print(f"{'Task':<45} {'Diff':6} {'Agg':>6} {'Steps':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['task_id']:<45} {r['difficulty']:6} "
              f"{r['grade_aggregate']:>6.3f} {r['steps']:>6}")
    print("-" * 70)
    mean_agg = sum(r["grade_aggregate"] for r in results) / len(results)
    print(f"{'MEAN AGGREGATE':>54} {mean_agg:>6.3f}")

    # Write JSON results
    out_path = os.path.join(os.path.dirname(__file__), "baseline_scores.json")
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "seed": args.seed, "results": results},
                  f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
