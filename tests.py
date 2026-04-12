"""
Transplant Logistics Environment — Test Suite
Validates environment mechanics, grader correctness, and task integrity.

Run:  python tests.py
      python -m pytest tests.py -v   (if pytest is installed)
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import (
    BLOOD_COMPAT, TASKS, TransplantEnv, TransplantGrader,
    VIABILITY_HOURS, compatibility_score, haversine_km,
    transport_minutes, _hospitals, NHSDataLoader,
)
from models import (
    ActionType, BloodType, Donor, Hospital, OrganType,
    Recipient, TransplantAction, TransportMode, UrgencyTier,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  ✅ {name}")
    else:
        _failed += 1
        msg = f"  ❌ {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TASK INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

def test_task_integrity():
    print("\n── 1. Task Integrity ──")
    check("At least 3 tasks defined", len(TASKS) >= 3, f"found {len(TASKS)}")
    check("5 tasks defined (above minimum)", len(TASKS) == 5, f"found {len(TASKS)}")

    expected = [
        "task_easy_clear_match",
        "task_medium_cascade_allocation",
        "task_hard_expiry_crisis",
        "task_medhard_dcd_split",
        "task_expert_national_surge",
    ]
    for tid in expected:
        check(f"Task '{tid}' exists", tid in TASKS)

    for tid, task in TASKS.items():
        check(f"{tid}: has donors", len(task["donors"]) > 0)
        check(f"{tid}: has recipients", len(task["recipients"]) > 0)
        check(f"{tid}: has max_steps", task["max_steps"] > 0)
        check(f"{tid}: has required_actions", len(task["required_actions"]) > 0)
        check(f"{tid}: has description", len(task["description"]) > 20)

    # Difficulty progression
    difficulties = [TASKS[t]["difficulty"] for t in expected]
    check(
        "Difficulty progression exists",
        len(set(difficulties)) >= 3,
        f"found: {difficulties}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BLOOD TYPE COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

def test_blood_compat():
    print("\n── 2. Blood Type Compatibility ──")
    # O- is universal donor
    check("O- donates to all 8 types", len(BLOOD_COMPAT[BloodType.O_NEG]) == 8)
    # AB+ only donates to AB+
    check("AB+ donates only to AB+", BLOOD_COMPAT[BloodType.AB_POS] == [BloodType.AB_POS])
    # A+ donates to A+ and AB+
    check("A+ donates to A+ and AB+",
          set(BLOOD_COMPAT[BloodType.A_POS]) == {BloodType.A_POS, BloodType.AB_POS})


# ═══════════════════════════════════════════════════════════════════════════════
# 3. COMPATIBILITY SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def test_compatibility_scoring():
    print("\n── 3. Compatibility Scoring ──")
    hosp_map = {h.hospital_id: h for h in _hospitals()}

    # Incompatible blood type → 0.0
    donor = Donor(
        donor_id="T1", organ_type=OrganType.KIDNEY, blood_type=BloodType.A_POS,
        age=40, hospital_id="H_GUY", procurement_time_utc="2026-01-01T00:00:00Z",
        viability_hours=24.0,
    )
    recip = Recipient(
        recipient_id="T2", organ_needed=OrganType.KIDNEY, blood_type=BloodType.B_POS,
        age=40, hospital_id="H_GUY", urgency=UrgencyTier.STATUS_2, wait_days=100,
    )
    score = compatibility_score(donor, recip, hosp_map)
    check("Incompatible blood → 0.0", score == 0.0, f"got {score}")

    # Organ mismatch → 0.0
    recip2 = Recipient(
        recipient_id="T3", organ_needed=OrganType.HEART, blood_type=BloodType.A_POS,
        age=40, hospital_id="H_GUY", urgency=UrgencyTier.STATUS_2, wait_days=100,
    )
    score2 = compatibility_score(donor, recip2, hosp_map)
    check("Organ mismatch → 0.0", score2 == 0.0, f"got {score2}")

    # Compatible match → positive score
    recip3 = Recipient(
        recipient_id="T4", organ_needed=OrganType.KIDNEY, blood_type=BloodType.A_POS,
        age=40, hospital_id="H_GUY", urgency=UrgencyTier.STATUS_1A, wait_days=300,
    )
    score3 = compatibility_score(donor, recip3, hosp_map)
    check("Compatible match → positive", 0.0 < score3 <= 1.0, f"got {score3}")

    # High PRA penalises score
    recip4 = Recipient(
        recipient_id="T5", organ_needed=OrganType.KIDNEY, blood_type=BloodType.A_POS,
        age=40, hospital_id="H_GUY", urgency=UrgencyTier.STATUS_1A, wait_days=300,
        pra=0.90,
    )
    score4 = compatibility_score(donor, recip4, hosp_map)
    check("High PRA lowers score", score4 < score3, f"PRA=0.90: {score4} vs base: {score3}")

    # Score always in [0, 1]
    check("Score in [0, 1] range", 0.0 <= score3 <= 1.0)
    check("Score in [0, 1] range (high PRA)", 0.0 <= score4 <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRANSPORT & GEOGRAPHY
# ═══════════════════════════════════════════════════════════════════════════════

def test_transport():
    print("\n── 4. Transport & Geography ──")
    hospitals = _hospitals()
    check("8 UK hospitals defined", len(hospitals) == 8)

    hosp_map = {h.hospital_id: h for h in hospitals}
    # London to Edinburgh should be ~530 km
    guys = hosp_map["H_GUY"]
    edi = hosp_map["H_ERI"]
    dist = haversine_km(guys.lat, guys.lon, edi.lat, edi.lon)
    check("London→Edinburgh ~530 km", 500 < dist < 560, f"got {dist:.0f} km")

    # Same hospital → very short ground time
    t_same = transport_minutes(guys, guys, TransportMode.GROUND)
    check("Same hospital ground ~15 min", 14 <= t_same <= 16, f"got {t_same:.1f} min")

    # Charter should be faster than ground for long distances
    t_ground = transport_minutes(guys, edi, TransportMode.GROUND)
    t_charter = transport_minutes(guys, edi, TransportMode.CHARTER)
    check(
        "Charter faster than ground (London→Edinburgh)",
        t_charter < t_ground,
        f"charter={t_charter:.0f} min, ground={t_ground:.0f} min",
    )

    # Leeds has no charter access
    leeds = hosp_map["H_LGI"]
    check("Leeds has no charter access", not leeds.has_charter_access)

    # All hospital IDs are UK format
    for h in hospitals:
        check(f"Hospital {h.hospital_id} has UK ID", h.hospital_id.startswith("H_"))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ENVIRONMENT RESET & STATE
# ═══════════════════════════════════════════════════════════════════════════════

def test_reset_and_state():
    print("\n── 5. Reset & State ──")
    for tid in TASKS:
        env = TransplantEnv(tid)
        obs = env.reset(seed=42)
        check(f"{tid}: reset returns observation", obs is not None)
        check(f"{tid}: step=0 after reset", obs.step == 0)
        check(f"{tid}: donors present", len(obs.available_donors) > 0)
        check(f"{tid}: waitlist present", len(obs.waitlist) > 0)
        check(f"{tid}: elapsed=0 after reset", obs.elapsed_minutes == 0.0)

        state = env.state()
        check(f"{tid}: state returns clean state", state.organs_wasted == 0)
        check(f"{tid}: state has empty action_log", len(state.action_log) == 0)

        # Reset again should produce identical state
        obs2 = env.reset(seed=42)
        check(f"{tid}: deterministic reset (same seed)", obs.step == obs2.step)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ACTION EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def test_actions():
    print("\n── 6. Action Execution ──")
    env = TransplantEnv("task_easy_clear_match")
    env.reset(seed=42)

    # Valid match: D001 → R001 (compatible)
    action = TransplantAction(
        action_type=ActionType.MATCH_ORGAN,
        donor_id="D001", recipient_id="R001",
        transport_mode=TransportMode.GROUND,
    )
    result = env.step(action)
    check("Valid match → positive reward", result.reward > 0, f"got {result.reward}")
    check("Valid match → not done yet", result.done is False)

    # Dispatch
    dispatch = TransplantAction(
        action_type=ActionType.DISPATCH_TRANSPORT,
        donor_id="D001",
        transport_mode=TransportMode.GROUND,
    )
    result2 = env.step(dispatch)
    check("Dispatch → positive reward", result2.reward > 0, f"got {result2.reward}")

    # Notify (ends episode)
    notify = TransplantAction(action_type=ActionType.NOTIFY_TEAM)
    result3 = env.step(notify)
    check("Notify → done=True", result3.done is True)
    check("Notify → positive reward", result3.reward > 0, f"got {result3.reward}")

    # Test incompatible match on fresh env
    env2 = TransplantEnv("task_easy_clear_match")
    env2.reset(seed=42)
    bad_match = TransplantAction(
        action_type=ActionType.MATCH_ORGAN,
        donor_id="D001", recipient_id="R003",  # B+ recipient, A+ donor → incompatible
    )
    bad_result = env2.step(bad_match)
    check("Incompatible match → negative reward", bad_result.reward < 0,
          f"got {bad_result.reward}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ORGAN EXPIRY
# ═══════════════════════════════════════════════════════════════════════════════

def test_organ_expiry():
    print("\n── 7. Organ Expiry ──")
    # The hard task has a lung with 3.5h viability → expires after 7 pass_to_next steps
    env = TransplantEnv("task_hard_expiry_crisis")
    env.reset(seed=42)

    # Pass 7 times (7 × 30 min = 210 min = 3.5h → lung expires)
    for i in range(7):
        result = env.step(TransplantAction(action_type=ActionType.PASS_TO_NEXT))

    state = env.state()
    check("Lung expired after 7 passes", state.organs_wasted >= 1,
          f"wasted={state.organs_wasted}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GRADER
# ═══════════════════════════════════════════════════════════════════════════════

def test_grader():
    print("\n── 8. Grader ──")
    grader = TransplantGrader()

    # Perfect easy run
    env = TransplantEnv("task_easy_clear_match")
    env.reset(seed=42)
    env.step(TransplantAction(action_type=ActionType.MATCH_ORGAN,
                               donor_id="D001", recipient_id="R001",
                               transport_mode=TransportMode.GROUND))
    env.step(TransplantAction(action_type=ActionType.DISPATCH_TRANSPORT,
                               donor_id="D001",
                               transport_mode=TransportMode.GROUND))
    env.step(TransplantAction(action_type=ActionType.NOTIFY_TEAM))

    state = env.state()
    task = TASKS["task_easy_clear_match"]
    grades = grader.grade(state, task)

    check("Grader returns aggregate", "aggregate" in grades)
    check("Aggregate in [0, 1]", 0.0 <= grades["aggregate"] <= 1.0,
          f"got {grades['aggregate']}")
    check("Perfect easy run scores high", grades["aggregate"] > 0.5,
          f"got {grades['aggregate']}")
    check("transplant_rate = 1.0 (matched all donors)",
          grades["transplant_rate"] == 1.0)
    check("safety = 1.0 (no dangerous matches)", grades["safety"] == 1.0)

    # Grader has 7 components
    expected_keys = [
        "transplant_rate", "quality", "viability_margin",
        "urgency_priority", "safety", "step_efficiency", "transport_optimality",
    ]
    for k in expected_keys:
        check(f"Grader returns '{k}'", k in grades, f"keys: {list(grades.keys())}")

    # Step efficiency: 3 steps for 3 required actions = 1.0
    check("Step efficiency = 1.0 (optimal steps)",
          grades["step_efficiency"] == 1.0,
          f"got {grades['step_efficiency']}")

    # Zero-action run should score low
    env2 = TransplantEnv("task_easy_clear_match")
    env2.reset(seed=42)
    env2.step(TransplantAction(action_type=ActionType.NOTIFY_TEAM))
    state2 = env2.state()
    grades2 = grader.grade(state2, task)
    check("Early notify scores low", grades2["aggregate"] < 0.5,
          f"got {grades2['aggregate']}")
    check("transplant_rate = 0 when no matches", grades2["transplant_rate"] == 0.0)

    # Grader is deterministic
    grades_again = grader.grade(state, task)
    check("Grader is deterministic", grades["aggregate"] == grades_again["aggregate"])


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ALL TASKS COMPLETE WITHOUT ERROR
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_tasks_run():
    print("\n── 9. All Tasks Run Without Error ──")
    grader = TransplantGrader()
    for tid in TASKS:
        try:
            env = TransplantEnv(tid)
            obs = env.reset(seed=42)
            # Take a few pass actions then notify
            for _ in range(3):
                result = env.step(TransplantAction(action_type=ActionType.PASS_TO_NEXT))
                if result.done:
                    break
            if not result.done:
                env.step(TransplantAction(action_type=ActionType.NOTIFY_TEAM))
            state = env.state()
            grades = grader.grade(state, TASKS[tid])
            check(f"{tid}: completes without error", True)
            check(f"{tid}: grade in [0, 1]",
                  0.0 <= grades["aggregate"] <= 1.0,
                  f"got {grades['aggregate']}")
        except Exception as e:
            check(f"{tid}: completes without error", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 10. NHS DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def test_nhs_data_loader():
    print("\n── 10. NHS Data Loader ──")
    loader = NHSDataLoader()
    check("NHSDataLoader instantiates without CSV", True)

    kidney = loader.get_organ_stats("kidney")
    check("Kidney stats available", len(kidney) > 0)
    check("Kidney annual_transplants = 2262", kidney["annual_transplants"] == 2262)

    check("Total deceased donors = 1430", loader.total_deceased_donors == 1430)
    check("Total transplants = 3479", loader.total_transplants == 3479)

    # Unknown organ returns empty
    unknown = loader.get_organ_stats("spleen")
    check("Unknown organ → empty dict", unknown == {})


# ═══════════════════════════════════════════════════════════════════════════════
# 11. DCD TASK SPECIFIC CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def test_dcd_task():
    print("\n── 11. DCD Task (medium-hard) ──")
    task = TASKS["task_medhard_dcd_split"]
    check("DCD task has 2 donors", len(task["donors"]) == 2)
    check("DCD task has 4 recipients", len(task["recipients"]) == 4)

    # Verify reduced DCD viability
    pancreas_donor = next(d for d in task["donors"] if d.organ_type == OrganType.PANCREAS)
    kidney_donor = next(d for d in task["donors"] if d.organ_type == OrganType.KIDNEY)
    check("DCD pancreas viability < standard 12h",
          pancreas_donor.viability_hours < 12.0,
          f"got {pancreas_donor.viability_hours}h")
    check("DCD kidney viability < standard 24h",
          kidney_donor.viability_hours < 24.0,
          f"got {kidney_donor.viability_hours}h")

    # High-PRA trap exists
    high_pra = [r for r in task["recipients"] if r.pra > 0.80]
    check("DCD task has high-PRA trap", len(high_pra) >= 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. EXPERT TASK SPECIFIC CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def test_expert_task():
    print("\n── 12. Expert Task (national surge) ──")
    task = TASKS["task_expert_national_surge"]
    check("Expert task has 4 donors", len(task["donors"]) == 4)
    check("Expert task has 8 recipients", len(task["recipients"]) == 8)

    # Donors across different hospitals
    donor_hospitals = set(d.hospital_id for d in task["donors"])
    check("Donors across 4 different hospitals", len(donor_hospitals) == 4,
          f"got {donor_hospitals}")

    # Contains all organ types in donors
    donor_organs = set(d.organ_type for d in task["donors"])
    check("All 4 major organ types present",
          donor_organs >= {OrganType.HEART, OrganType.LIVER, OrganType.LUNG, OrganType.KIDNEY},
          f"got {donor_organs}")

    # Has tight viability lung
    lung = next(d for d in task["donors"] if d.organ_type == OrganType.LUNG)
    check("Lung has tight viability (≤5h)", lung.viability_hours <= 5.0,
          f"got {lung.viability_hours}h")

    # max_steps is generous enough
    check("Expert max_steps ≥ 20", task["max_steps"] >= 20)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("TRANSPLANT LOGISTICS — TEST SUITE")
    print("=" * 60)

    test_task_integrity()
    test_blood_compat()
    test_compatibility_scoring()
    test_transport()
    test_reset_and_state()
    test_actions()
    test_organ_expiry()
    test_grader()
    test_all_tasks_run()
    test_nhs_data_loader()
    test_dcd_task()
    test_expert_task()

    print("\n" + "=" * 60)
    print(f"RESULTS: {_passed} passed, {_failed} failed")
    print("=" * 60)

    if _failed > 0:
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS PASSED")


if __name__ == "__main__":
    main()
