"""
NHS Organ Donation Data Explorer + Smoke Test
=============================================
Inspects the Kaggle NHS dataset (patricklford/nhs-organ-donation)
and runs a heuristic agent through all three tasks to verify the
full pipeline is wired up correctly.

Usage:
    # Without CSV (uses fallback NHS 2022/23 stats)
    python nhs_data_explorer.py

    # With the Kaggle CSV file or directory
    python nhs_data_explorer.py --csv data/nhs/
    python nhs_data_explorer.py --csv data/nhs/nhs_organ_donation.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import NHSDataLoader, TASKS, TransplantEnv, TransplantGrader
from models import ActionType, TransplantAction, TransportMode


# ── Data inspection ────────────────────────────────────────────────────────────

def find_csv(path: str) -> str | None:
    """Resolve a CSV path — accepts either a file or a directory."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = glob.glob(os.path.join(path, "*.csv"))
        if candidates:
            return candidates[0]
    return None


def print_nhs_stats(loader: NHSDataLoader) -> None:
    organs = ["kidney", "liver", "heart", "lung", "pancreas"]
    print("\n" + "═" * 60)
    print("NHS Blood and Transplant — Calibration Statistics")
    print("═" * 60)
    for organ in organs:
        s = loader.get_organ_stats(organ)
        if not s:
            continue
        print(f"\n  {organ.upper()}")
        print(f"    Annual transplants  : {s.get('annual_transplants', 'N/A')}")
        print(f"    Waiting list size   : {s.get('waiting_list_size',  'N/A')}")
        print(f"    Median wait (days)  : {s.get('median_wait_days',   'N/A')}")
        util = s.get('utilisation_rate')
        if util is not None:
            print(f"    Utilisation rate    : {util:.0%}")
        pct_dcd = s.get('pct_dcd')
        if pct_dcd is not None:
            print(f"    % DCD donors        : {pct_dcd:.0%}")
    print()


# ── Heuristic agent ────────────────────────────────────────────────────────────

def heuristic_action(obs) -> TransplantAction:
    """
    Simple rule-based agent used for smoke testing.
    Priority: crossmatch high-PRA → match best compatible pair → dispatch → notify.
    """
    # If any pending match is accepted and not yet dispatched, dispatch it
    for m in obs.pending_matches:
        if m.accepted:
            return TransplantAction(
                action_type=ActionType.DISPATCH_TRANSPORT,
                donor_id=m.donor_id,
                transport_mode=TransportMode.CHARTER,
            )

    # Request crossmatch for any high-PRA recipient paired with a donor
    for donor in obs.available_donors:
        for recip in obs.waitlist:
            if (recip.organ_needed.value == donor.organ_type.value
                    and recip.pra > 0.85
                    and recip.blood_type == donor.blood_type):
                already = any(
                    m.donor_id == donor.donor_id
                    and m.recipient_id == recip.recipient_id
                    for m in obs.pending_matches
                )
                if not already:
                    return TransplantAction(
                        action_type=ActionType.REQUEST_CROSSMATCH,
                        donor_id=donor.donor_id,
                        recipient_id=recip.recipient_id,
                    )

    # Match best compatible pair (greedily: urgency 1A first, skip high-PRA without XM)
    best_action = None
    best_priority = -1
    for donor in obs.available_donors:
        for recip in obs.waitlist:
            if recip.organ_needed.value != donor.organ_type.value:
                continue
            # Skip dangerous high-PRA without crossmatch
            if recip.pra > 0.85:
                xm_done = any(
                    m.donor_id == donor.donor_id
                    and m.recipient_id == recip.recipient_id
                    and m.crossmatch_pending
                    for m in obs.pending_matches
                )
                if not xm_done:
                    continue
            urgency_score = {"1A": 3, "1B": 2, "2": 1, "7": 0}
            priority = urgency_score.get(recip.urgency.value, 0)
            if priority > best_priority:
                best_priority = priority
                best_action   = TransplantAction(
                    action_type=ActionType.MATCH_ORGAN,
                    donor_id=donor.donor_id,
                    recipient_id=recip.recipient_id,
                    transport_mode=TransportMode.CHARTER,
                )

    if best_action:
        return best_action

    # No more donors or recipients — notify teams
    return TransplantAction(action_type=ActionType.NOTIFY_TEAM)


# ── Smoke test ─────────────────────────────────────────────────────────────────

def run_smoke_test(csv_path: str | None = None) -> None:
    grader = TransplantGrader()
    results = []

    for task_id, task_meta in TASKS.items():
        print(f"\n{'─' * 60}")
        print(f"Smoke test: {task_meta['name']}")
        print(f"Difficulty: {task_meta['difficulty']}  |  Max steps: {task_meta['max_steps']}")
        print(f"{'─' * 60}")

        env    = TransplantEnv(task_id, nhs_csv_path=csv_path)
        obs    = env.reset(seed=42)
        done   = False
        step   = 0
        total_r = 0.0

        while not done:
            action = heuristic_action(obs)
            result = env.step(action)
            obs    = result.observation
            done   = result.done
            total_r += result.reward
            step   += 1

            status = "✓"
            if result.info.get("error"):
                status = f"✗  {result.info['error']}"
            elif result.info.get("warning"):
                status = f"⚠  {result.info['warning'][:60]}"

            print(f"  Step {step:02d} | {action.action_type.value:<22} "
                  f"reward={result.reward:+.3f}  {status}")

            if step >= task_meta["max_steps"]:
                break

        final  = env.state()
        grades = grader.grade(final, TASKS[task_id])
        print(f"\n  Aggregate score: {grades['aggregate']:.3f}  "
              f"(transplant_rate={grades['transplant_rate']:.2f}, "
              f"quality={grades['quality']:.2f}, "
              f"safety={grades['safety']:.2f})")

        results.append((task_id, task_meta["difficulty"], grades["aggregate"]))

    print(f"\n{'═' * 60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'═' * 60}")
    print(f"{'Task':<40} {'Diff':8} {'Score':>6}")
    print(f"{'─' * 60}")
    for tid, diff, score in results:
        ok = "✓" if score > 0.0 else "✗"
        print(f"  {ok}  {tid:<38} {diff:<8} {score:.3f}")
    print(f"{'─' * 60}")
    all_ok = all(s > 0.0 for _, _, s in results)
    print(f"\n  Pipeline status: {'✓ ALL TASKS PASSED' if all_ok else '✗ SOME TASKS FAILED'}")
    print(f"  (Heuristic agent — not optimised; scores reflect rule-based baseline)\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inspect NHS organ donation data and smoke-test the environment."
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to NHS Kaggle CSV file or directory containing it. "
             "If omitted, fallback NHS 2022/23 statistics are used.",
    )
    args = parser.parse_args()

    csv_path = None
    if args.csv:
        csv_path = find_csv(args.csv)
        if csv_path:
            print(f"[explorer] Using CSV: {csv_path}")
        else:
            print(f"[explorer] No CSV found at '{args.csv}' — using fallback stats.")

    loader = NHSDataLoader(csv_path)
    print_nhs_stats(loader)
    run_smoke_test(csv_path)


if __name__ == "__main__":
    main()
