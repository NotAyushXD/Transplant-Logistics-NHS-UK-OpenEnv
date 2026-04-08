"""
Transplant Logistics Environment — Core Logic
Simulates real-world organ allocation decisions:
  - Donor ↔ recipient compatibility matching
  - Cold ischaemia viability windows
  - Transport logistics
  - UNOS-style priority scoring

Three tasks: easy (single organ, clear match) →
             medium (multi-organ, competing recipients) →
             hard (cascade failure, expiring viability, misleading signals)
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ActionType, BloodType, Donor, Hospital, MatchRecord, OrganType,
    Recipient, StepResult, TransplantAction, TransplantObservation,
    TransplantState, TransportLeg, TransportMode, UrgencyTier,
)

# ── Blood type compatibility table ───────────────────────────────────────────
# donor → set of compatible recipient blood types
BLOOD_COMPAT: Dict[BloodType, List[BloodType]] = {
    BloodType.O_NEG:  list(BloodType),                          # universal donor
    BloodType.O_POS:  [BloodType.O_POS, BloodType.A_POS,
                       BloodType.B_POS, BloodType.AB_POS],
    BloodType.A_NEG:  [BloodType.A_NEG, BloodType.A_POS,
                       BloodType.AB_NEG, BloodType.AB_POS],
    BloodType.A_POS:  [BloodType.A_POS, BloodType.AB_POS],
    BloodType.B_NEG:  [BloodType.B_NEG, BloodType.B_POS,
                       BloodType.AB_NEG, BloodType.AB_POS],
    BloodType.B_POS:  [BloodType.B_POS, BloodType.AB_POS],
    BloodType.AB_NEG: [BloodType.AB_NEG, BloodType.AB_POS],
    BloodType.AB_POS: [BloodType.AB_POS],
}

# Viability windows by organ (hours)
VIABILITY_HOURS: Dict[OrganType, float] = {
    OrganType.HEART:   4.0,
    OrganType.LUNG:    6.0,
    OrganType.LIVER:   12.0,
    OrganType.KIDNEY:  24.0,
    OrganType.PANCREAS: 12.0,
}

# Minutes per step
MINUTES_PER_STEP = 30


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def transport_minutes(h1: Hospital, h2: Hospital,
                      mode: TransportMode) -> float:
    dist = haversine_km(h1.lat, h1.lon, h2.lat, h2.lon)
    speeds = {
        TransportMode.GROUND:     60.0,   # km/h
        TransportMode.CHARTER:   500.0,
        TransportMode.COMMERCIAL: 700.0,
    }
    overhead = {
        TransportMode.GROUND:      15,
        TransportMode.CHARTER:     90,
        TransportMode.COMMERCIAL: 180,
    }
    return (dist / speeds[mode]) * 60 + overhead[mode]


# ── Compatibility scoring ─────────────────────────────────────────────────────

def compatibility_score(donor: Donor, recipient: Recipient,
                        hosp_map: Dict[str, Hospital]) -> float:
    """Returns 0.0–1.0. 0.0 = incompatible."""
    # Blood type
    if recipient.blood_type not in BLOOD_COMPAT.get(donor.blood_type, []):
        return 0.0
    # Organ match
    if donor.organ_type != recipient.organ_needed:
        return 0.0

    score = 0.5  # base for blood+organ match

    # HLA antibody penalty
    crossreact = set(donor.hla_antigens) & set(recipient.hla_antibodies)
    hla_penalty = len(crossreact) * 0.08
    score -= hla_penalty

    # PRA penalty
    score -= recipient.pra * 0.15

    # Urgency bonus
    urgency_bonus = {"1A": 0.25, "1B": 0.15, "2": 0.05, "7": 0.0}
    score += urgency_bonus.get(recipient.urgency.value, 0.0)

    # Wait time bonus (log-scaled)
    score += min(0.15, math.log1p(recipient.wait_days) / 40)

    # Age mismatch penalty for kidneys
    if donor.organ_type == OrganType.KIDNEY:
        age_diff = abs(donor.age - recipient.age)
        score -= min(0.10, age_diff * 0.002)

    # KDPI for kidney (lower = better donor)
    if donor.kdpi is not None and donor.organ_type == OrganType.KIDNEY:
        score += (1 - donor.kdpi) * 0.10

    # MELD for liver
    if donor.organ_type == OrganType.LIVER and recipient.meld_score:
        score += min(0.10, recipient.meld_score / 400)

    return max(0.0, min(1.0, round(score, 3)))


# ── Task definitions ──────────────────────────────────────────────────────────

def _hospitals() -> List[Hospital]:
    return [
        Hospital(hospital_id="H_MUM", name="Mumbai PGIMS",      lat=19.076, lon=72.877, has_charter_access=True),
        Hospital(hospital_id="H_DEL", name="AIIMS Delhi",        lat=28.567, lon=77.210, has_charter_access=True),
        Hospital(hospital_id="H_BLR", name="Manipal Bengaluru",  lat=12.972, lon=77.594, has_charter_access=True),
        Hospital(hospital_id="H_CHE", name="Apollo Chennai",     lat=13.082, lon=80.270, has_charter_access=True),
        Hospital(hospital_id="H_HYD", name="KIMS Hyderabad",     lat=17.385, lon=78.487, has_charter_access=True),
        Hospital(hospital_id="H_KOL", name="SSKM Kolkata",       lat=22.572, lon=88.363, has_charter_access=False),
        Hospital(hospital_id="H_PUN", name="Ruby Hall Pune",     lat=18.520, lon=73.856, has_charter_access=True),
        Hospital(hospital_id="H_AHM", name="Civil Ahmedabad",    lat=23.022, lon=72.571, has_charter_access=True),
    ]


TASKS: Dict[str, Dict] = {

    # ── EASY: one donor, one clearly correct recipient ─────────────────────
    "task_easy_clear_match": {
        "id": "task_easy_clear_match",
        "name": "Single kidney — clear best match",
        "difficulty": "easy",
        "description": (
            "A deceased donor kidney is available in Mumbai. "
            "Three recipients are waitlisted; one is the clear best match "
            "(compatible blood type, Status 1A, low PRA, same city). "
            "Match, dispatch transport, and notify the surgical team."
        ),
        "max_steps": 8,
        "donors": [
            Donor(donor_id="D001", organ_type=OrganType.KIDNEY,
                  blood_type=BloodType.A_POS, age=42,
                  hospital_id="H_MUM",
                  procurement_time_utc="2026-04-25T06:00:00Z",
                  viability_hours=24.0, hla_antigens=["A2", "B44"],
                  kdpi=0.35),
        ],
        "recipients": [
            Recipient(recipient_id="R001", organ_needed=OrganType.KIDNEY,
                      blood_type=BloodType.A_POS, age=38,
                      hospital_id="H_MUM", urgency=UrgencyTier.STATUS_1A,
                      wait_days=420, hla_antibodies=[], pra=0.05),
            Recipient(recipient_id="R002", organ_needed=OrganType.KIDNEY,
                      blood_type=BloodType.AB_POS, age=55,
                      hospital_id="H_DEL", urgency=UrgencyTier.STATUS_2,
                      wait_days=120, hla_antibodies=["A2"], pra=0.45),
            Recipient(recipient_id="R003", organ_needed=OrganType.KIDNEY,
                      blood_type=BloodType.B_POS, age=40,   # incompatible
                      hospital_id="H_BLR", urgency=UrgencyTier.STATUS_1B,
                      wait_days=200, hla_antibodies=[], pra=0.10),
        ],
        "required_actions": [
            ActionType.MATCH_ORGAN,
            ActionType.DISPATCH_TRANSPORT,
            ActionType.NOTIFY_TEAM,
        ],
    },

    # ── MEDIUM: multi-organ, competing urgencies, time pressure ───────────
    "task_medium_cascade_allocation": {
        "id": "task_medium_cascade_allocation",
        "name": "Multi-organ cascade — liver + heart under viability pressure",
        "difficulty": "medium",
        "description": (
            "Two organs from the same donor: a liver (12 h viability) and "
            "a heart (4 h viability). Multiple recipients across cities. "
            "The heart must be matched and transported first or it expires. "
            "One recipient has a high PRA that makes them unsuitable despite "
            "high urgency. Allocate both organs correctly before time runs out."
        ),
        "max_steps": 14,
        "donors": [
            Donor(donor_id="D002", organ_type=OrganType.HEART,
                  blood_type=BloodType.O_POS, age=28,
                  hospital_id="H_BLR",
                  procurement_time_utc="2026-04-25T08:00:00Z",
                  viability_hours=4.0, hla_antigens=["B7"]),
            Donor(donor_id="D003", organ_type=OrganType.LIVER,
                  blood_type=BloodType.O_POS, age=28,
                  hospital_id="H_BLR",
                  procurement_time_utc="2026-04-25T08:00:00Z",
                  viability_hours=12.0, hla_antigens=["B7"],
                  meld_score=None),
        ],
        "recipients": [
            # Heart recipients
            Recipient(recipient_id="R010", organ_needed=OrganType.HEART,
                      blood_type=BloodType.O_POS, age=35,
                      hospital_id="H_BLR", urgency=UrgencyTier.STATUS_1A,
                      wait_days=30, hla_antibodies=[], pra=0.05),
            Recipient(recipient_id="R011", organ_needed=OrganType.HEART,
                      blood_type=BloodType.A_POS, age=40,    # incompatible
                      hospital_id="H_HYD", urgency=UrgencyTier.STATUS_1A,
                      wait_days=60, hla_antibodies=[], pra=0.10),
            Recipient(recipient_id="R012", organ_needed=OrganType.HEART,
                      blood_type=BloodType.O_NEG, age=50,
                      hospital_id="H_CHE", urgency=UrgencyTier.STATUS_1B,
                      wait_days=90, hla_antibodies=["B7"], pra=0.80),  # high PRA
            # Liver recipients
            Recipient(recipient_id="R020", organ_needed=OrganType.LIVER,
                      blood_type=BloodType.O_POS, age=48,
                      hospital_id="H_HYD", urgency=UrgencyTier.STATUS_1A,
                      wait_days=180, hla_antibodies=[], pra=0.10,
                      meld_score=32),
            Recipient(recipient_id="R021", organ_needed=OrganType.LIVER,
                      blood_type=BloodType.B_POS, age=52,    # incompatible
                      hospital_id="H_MUM", urgency=UrgencyTier.STATUS_1B,
                      wait_days=300, hla_antibodies=[], pra=0.05,
                      meld_score=28),
        ],
        "required_actions": [
            ActionType.MATCH_ORGAN,   # heart first
            ActionType.DISPATCH_TRANSPORT,
            ActionType.MATCH_ORGAN,   # then liver
            ActionType.DISPATCH_TRANSPORT,
            ActionType.NOTIFY_TEAM,
        ],
    },

    # ── HARD: expiring organs, false urgency signal, crossmatch needed ────
    "task_hard_expiry_crisis": {
        "id": "task_hard_expiry_crisis",
        "name": "Expiry crisis — misleading urgency + crossmatch hold",
        "difficulty": "hard",
        "description": (
            "Three donors, five recipients, limited viability. "
            "The most urgent recipient (Status 1A) has a high PRA and a "
            "positive virtual crossmatch — transplanting them risks hyperacute "
            "rejection. A second recipient appears lower urgency but is the "
            "only safe match for the heart. The kidney donor has an elevated "
            "KDPI (0.82) — it should be offered to older recipients. "
            "One lung is nearing expiry and must be matched within 3 steps "
            "or it's wasted. Correct allocation requires ignoring the "
            "urgency shortcut and reasoning about compatibility."
        ),
        "max_steps": 20,
        "donors": [
            Donor(donor_id="D101", organ_type=OrganType.LUNG,
                  blood_type=BloodType.A_NEG, age=45,
                  hospital_id="H_DEL",
                  procurement_time_utc="2026-04-25T10:00:00Z",
                  viability_hours=3.5,     # almost expired at start
                  hla_antigens=["A1", "B8"]),
            Donor(donor_id="D102", organ_type=OrganType.HEART,
                  blood_type=BloodType.O_NEG, age=22,
                  hospital_id="H_DEL",
                  procurement_time_utc="2026-04-25T10:00:00Z",
                  viability_hours=4.0, hla_antigens=["A3"]),
            Donor(donor_id="D103", organ_type=OrganType.KIDNEY,
                  blood_type=BloodType.B_POS, age=68,
                  hospital_id="H_KOL",
                  procurement_time_utc="2026-04-25T10:00:00Z",
                  viability_hours=24.0, hla_antigens=["B35"],
                  kdpi=0.82),
        ],
        "recipients": [
            # Lung recipients
            Recipient(recipient_id="R101", organ_needed=OrganType.LUNG,
                      blood_type=BloodType.A_NEG, age=38,
                      hospital_id="H_DEL", urgency=UrgencyTier.STATUS_1A,
                      wait_days=90, hla_antibodies=["A1", "B8"],
                      pra=0.92),   # positive virtual crossmatch — DO NOT USE
            Recipient(recipient_id="R102", organ_needed=OrganType.LUNG,
                      blood_type=BloodType.A_POS, age=44,  # incompatible
                      hospital_id="H_BLR", urgency=UrgencyTier.STATUS_1B,
                      wait_days=210, hla_antibodies=[], pra=0.05),
            Recipient(recipient_id="R103", organ_needed=OrganType.LUNG,
                      blood_type=BloodType.A_NEG, age=50,
                      hospital_id="H_DEL", urgency=UrgencyTier.STATUS_1B,
                      wait_days=300, hla_antibodies=[], pra=0.08),  # correct match
            # Heart recipient
            Recipient(recipient_id="R110", organ_needed=OrganType.HEART,
                      blood_type=BloodType.O_NEG, age=29,
                      hospital_id="H_DEL", urgency=UrgencyTier.STATUS_1A,
                      wait_days=14, hla_antibodies=["A3"], pra=0.85),  # crossmatch risk
            Recipient(recipient_id="R111", organ_needed=OrganType.HEART,
                      blood_type=BloodType.O_POS, age=33,
                      hospital_id="H_DEL", urgency=UrgencyTier.STATUS_1B,
                      wait_days=60, hla_antibodies=[], pra=0.10),   # correct
            # Kidney recipient
            Recipient(recipient_id="R120", organ_needed=OrganType.KIDNEY,
                      blood_type=BloodType.B_POS, age=65,
                      hospital_id="H_KOL", urgency=UrgencyTier.STATUS_2,
                      wait_days=600, hla_antibodies=[], pra=0.05,
                      eGFR=12.0),  # correct — older, KDPI match
        ],
        "required_actions": [
            ActionType.REQUEST_CROSSMATCH,
            ActionType.MATCH_ORGAN,
            ActionType.DISPATCH_TRANSPORT,
            ActionType.MATCH_ORGAN,
            ActionType.DISPATCH_TRANSPORT,
            ActionType.MATCH_ORGAN,
            ActionType.DISPATCH_TRANSPORT,
            ActionType.NOTIFY_TEAM,
        ],
    },
}


# ── Environment ───────────────────────────────────────────────────────────────

class TransplantEnv:
    """
    OpenEnv-compliant transplant logistics environment.
    Implements step() / reset() / state() with typed Pydantic contracts.
    """

    def __init__(self, task_id: str):
        assert task_id in TASKS, f"Unknown task: {task_id}"
        self.task_id   = task_id
        self._task     = TASKS[task_id]
        self._hosp_map: Dict[str, Hospital] = {h.hospital_id: h for h in _hospitals()}
        self._state: Optional[TransplantState] = None

    # ── OpenEnv API ───────────────────────────────────────────────────────

    def reset(self, seed: int = 42) -> TransplantObservation:
        random.seed(seed)
        task = self._task
        self._state = TransplantState(
            task_id    = self.task_id,
            step       = 0,
            donors     = copy.deepcopy(task["donors"]),
            recipients = copy.deepcopy(task["recipients"]),
            hospitals  = list(self._hosp_map.values()),
            matches    = [],
            elapsed_minutes = 0.0,
            successful_transplants = 0,
            organs_wasted = 0,
            avg_wait_reduction_days = 0.0,
            action_log = [],
        )
        return self._make_observation()

    def step(self, action: TransplantAction) -> StepResult:
        assert self._state is not None, "Call reset() first"
        s = self._state
        s.step += 1
        s.elapsed_minutes += MINUTES_PER_STEP

        # Log action
        s.action_log.append({
            "step": s.step,
            "action": action.action_type.value,
            "donor_id": action.donor_id,
            "recipient_id": action.recipient_id,
        })

        # Advance cold ischaemia
        for donor in s.donors:
            donor.cross_clamp_time_minutes += MINUTES_PER_STEP

        reward, info = self._execute(action)

        # Check organ expiry
        expired = []
        for donor in s.donors:
            elapsed_h = donor.cross_clamp_time_minutes / 60
            if elapsed_h >= donor.viability_hours:
                if not any(m.donor_id == donor.donor_id and m.accepted
                           for m in s.matches):
                    s.organs_wasted += 1
                    info["alerts"] = info.get("alerts", []) + \
                        [f"ORGAN EXPIRED: {donor.organ_type.value} from {donor.donor_id}"]
                expired.append(donor.donor_id)
        s.donors = [d for d in s.donors if d.donor_id not in expired]

        done = (
            s.step >= self._task["max_steps"]
            or len(s.donors) == 0
            or action.action_type == ActionType.NOTIFY_TEAM
        )

        obs = self._make_observation()
        return StepResult(observation=obs, reward=round(reward, 4),
                          done=done, info=info)

    def state(self) -> TransplantState:
        assert self._state is not None, "Call reset() first"
        return copy.deepcopy(self._state)

    # ── Action execution ──────────────────────────────────────────────────

    def _execute(self, action: TransplantAction) -> Tuple[float, Dict]:
        s = self._state
        a = action.action_type
        info: Dict[str, Any] = {"action": a.value, "alerts": []}

        if a == ActionType.MATCH_ORGAN:
            return self._do_match(action, info)

        elif a == ActionType.DISPATCH_TRANSPORT:
            return self._do_dispatch(action, info)

        elif a == ActionType.REQUEST_CROSSMATCH:
            return self._do_crossmatch(action, info)

        elif a == ActionType.NOTIFY_TEAM:
            return self._do_notify(action, info)

        elif a == ActionType.DECLINE_ORGAN:
            return self._do_decline(action, info)

        elif a == ActionType.REJECT_MATCH:
            return self._do_reject(action, info)

        elif a == ActionType.PASS_TO_NEXT:
            # Deliberately skipping a candidate — small neutral reward
            info["message"] = "Passed to next candidate"
            return 0.0, info

        return 0.0, info

    def _do_match(self, action: TransplantAction,
                  info: Dict) -> Tuple[float, Dict]:
        s = self._state
        donor = next((d for d in s.donors if d.donor_id == action.donor_id), None)
        recip = next((r for r in s.recipients
                      if r.recipient_id == action.recipient_id), None)

        if donor is None:
            info["error"] = f"Donor {action.donor_id} not found"
            return -0.1, info
        if recip is None:
            info["error"] = f"Recipient {action.recipient_id} not found"
            return -0.1, info

        score = compatibility_score(donor, recip, self._hosp_map)
        if score == 0.0:
            info["error"] = "Incompatible match — blood type or organ mismatch"
            return -0.3, info

        # Check for dangerous high-PRA without crossmatch
        if recip.pra > 0.80:
            pending_xm = any(
                m.donor_id == donor.donor_id
                and m.recipient_id == recip.recipient_id
                and m.crossmatch_pending
                for m in s.matches
            )
            if not pending_xm:
                info["warning"] = (
                    f"Recipient {recip.recipient_id} has PRA={recip.pra:.0%}. "
                    "Crossmatch not requested — hyperacute rejection risk."
                )
                return -0.2, info

        # Calculate transport time
        dh = self._hosp_map[donor.hospital_id]
        rh = self._hosp_map[recip.hospital_id]
        mode = action.transport_mode or TransportMode.CHARTER
        t_min = transport_minutes(dh, rh, mode)
        remaining = (donor.viability_hours * 60) - donor.cross_clamp_time_minutes

        if t_min >= remaining:
            info["warning"] = (
                f"Transport time {t_min:.0f} min exceeds remaining viability "
                f"{remaining:.0f} min — organ will expire in transit."
            )
            return -0.25, info

        match = MatchRecord(
            donor_id=donor.donor_id,
            recipient_id=recip.recipient_id,
            compatibility_score=score,
            transport_minutes=t_min,
            remaining_viability_minutes=remaining,
            accepted=True,
        )
        s.matches.append(match)
        s.successful_transplants += 1
        s.recipients = [r for r in s.recipients
                        if r.recipient_id != recip.recipient_id]

        # Reward: scaled by score, urgency, and viability margin
        viability_margin = min(1.0, (remaining - t_min) / remaining)
        urgency_mult = {"1A": 1.5, "1B": 1.2, "2": 1.0, "7": 0.5}
        reward = (
            score * 0.5
            + viability_margin * 0.3
            + urgency_mult.get(recip.urgency.value, 1.0) * 0.2
        )
        info["match"] = match.model_dump()
        info["compatibility_score"] = score
        return round(reward, 4), info

    def _do_dispatch(self, action: TransplantAction,
                     info: Dict) -> Tuple[float, Dict]:
        s = self._state
        match = next((m for m in s.matches
                      if m.donor_id == action.donor_id and m.accepted), None)
        if match is None:
            info["error"] = "No accepted match found — match organ first"
            return -0.1, info

        mode = action.transport_mode or TransportMode.CHARTER
        donor = next((d for d in s.donors
                      if d.donor_id == action.donor_id), None)
        if donor is None:
            info["message"] = "Organ already dispatched"
            return 0.05, info

        dh = self._hosp_map.get(donor.hospital_id)
        # find recipient hospital
        recip_id = match.recipient_id
        recip_hosp = None
        for r_orig in self._task["recipients"]:
            if r_orig.recipient_id == recip_id:
                recip_hosp = self._hosp_map.get(r_orig.hospital_id)
                break

        if dh and recip_hosp:
            t_min = transport_minutes(dh, recip_hosp, mode)
            info["transport_dispatched"] = {
                "from": dh.name, "to": recip_hosp.name,
                "mode": mode.value, "eta_minutes": round(t_min),
            }

        s.donors = [d for d in s.donors if d.donor_id != action.donor_id]
        return 0.15, info

    def _do_crossmatch(self, action: TransplantAction,
                       info: Dict) -> Tuple[float, Dict]:
        s = self._state
        donor = next((d for d in s.donors
                      if d.donor_id == action.donor_id), None)
        recip = next((r for r in s.recipients
                      if r.recipient_id == action.recipient_id), None)
        if not donor or not recip:
            info["error"] = "Donor or recipient not found"
            return -0.05, info

        # Virtual crossmatch result
        crossreact = set(donor.hla_antigens) & set(recip.hla_antibodies)
        positive = bool(crossreact) or recip.pra > 0.85
        existing = next(
            (m for m in s.matches if m.donor_id == action.donor_id
             and m.recipient_id == action.recipient_id), None
        )
        if not existing:
            s.matches.append(MatchRecord(
                donor_id=donor.donor_id,
                recipient_id=recip.recipient_id,
                compatibility_score=0.0,
                transport_minutes=0.0,
                remaining_viability_minutes=0.0,
                crossmatch_pending=True,
            ))
        info["crossmatch_result"] = {
            "positive": positive,
            "crossreacting_antigens": list(crossreact),
            "pra": recip.pra,
            "recommendation": (
                "DO NOT PROCEED — hyperacute rejection risk"
                if positive else "Proceed with transplant"
            ),
        }
        # Reward for correct crossmatch request on high-PRA recipient
        reward = 0.10 if recip.pra > 0.50 else 0.0
        return reward, info

    def _do_notify(self, action: TransplantAction,
                   info: Dict) -> Tuple[float, Dict]:
        s = self._state
        accepted = [m for m in s.matches if m.accepted]
        info["notification_sent"] = True
        info["transplants_coordinated"] = len(accepted)
        # Bonus for completing all possible matches
        total_donors_orig = len(self._task["donors"])
        completion = len(accepted) / max(total_donors_orig, 1)
        return 0.1 + completion * 0.2, info

    def _do_decline(self, action: TransplantAction,
                    info: Dict) -> Tuple[float, Dict]:
        s = self._state
        # Declining a clearly incompatible match is slightly positive
        donor = next((d for d in s.donors
                      if d.donor_id == action.donor_id), None)
        recip = next((r for r in s.recipients
                      if r.recipient_id == action.recipient_id), None)
        if donor and recip:
            score = compatibility_score(donor, recip, self._hosp_map)
            if score == 0.0:
                info["message"] = "Correct decline — incompatible pair"
                return 0.05, info
        info["message"] = "Declined a potentially viable match"
        return -0.05, info

    def _do_reject(self, action: TransplantAction,
                   info: Dict) -> Tuple[float, Dict]:
        s = self._state
        s.matches = [m for m in s.matches
                     if not (m.donor_id == action.donor_id
                             and m.recipient_id == action.recipient_id)]
        info["message"] = f"Match rejected: {action.reason}"
        return 0.0, info

    # ── Observation builder ───────────────────────────────────────────────

    def _make_observation(self) -> TransplantObservation:
        s = self._state
        alerts = []
        for d in s.donors:
            remaining_h = (d.viability_hours * 60 - d.cross_clamp_time_minutes) / 60
            if remaining_h < 2.0:
                alerts.append(
                    f"URGENT: {d.organ_type.value} {d.donor_id} expires in "
                    f"{remaining_h:.1f}h"
                )
        return TransplantObservation(
            step=s.step,
            available_donors=s.donors,
            waitlist=s.recipients,
            hospitals=list(self._hosp_map.values()),
            pending_matches=s.matches,
            elapsed_minutes=s.elapsed_minutes,
            alerts=alerts,
            task_id=self.task_id,
            task_description=self._task["description"],
        )


# ── Grader ────────────────────────────────────────────────────────────────────

class TransplantGrader:
    """
    Scores a completed episode 0.0–1.0.
    Components:
      - transplant_rate:    fraction of available organs matched (0.35)
      - quality:            mean compatibility score of accepted matches (0.25)
      - viability_margin:   fraction of transplants with ≥ 30 min margin (0.20)
      - urgency_priority:   did highest-urgency recipients get matched first (0.10)
      - no_harmful_matches: penalty for dangerous high-PRA without crossmatch (0.10)
    """

    WEIGHTS = {
        "transplant_rate":   0.35,
        "quality":           0.25,
        "viability_margin":  0.20,
        "urgency_priority":  0.10,
        "safety":            0.10,
    }

    def grade(self, final_state: TransplantState,
              task: Dict) -> Dict[str, float]:
        s = final_state
        accepted = [m for m in s.matches if m.accepted]
        total_donors = len(task["donors"])

        # Transplant rate
        transplant_rate = len(accepted) / max(total_donors, 1)

        # Match quality
        quality = (
            sum(m.compatibility_score for m in accepted) / len(accepted)
            if accepted else 0.0
        )

        # Viability margin
        with_margin = [
            m for m in accepted
            if (m.remaining_viability_minutes - m.transport_minutes) >= 30
        ]
        viability_margin = len(with_margin) / max(len(accepted), 1) if accepted else 0.0

        # Urgency priority: did Status 1A recipients get matched?
        orig_1a = [r for r in task["recipients"]
                   if r.urgency == UrgencyTier.STATUS_1A]
        matched_1a = [
            m for m in accepted
            if any(r.recipient_id == m.recipient_id
                   and r.urgency == UrgencyTier.STATUS_1A
                   for r in task["recipients"])
        ]
        urgency_priority = (
            len(matched_1a) / len(orig_1a) if orig_1a else 1.0
        )

        # Safety: penalise dangerous high-PRA matches without crossmatch
        dangerous = 0
        for m in accepted:
            recip = next(
                (r for r in task["recipients"]
                 if r.recipient_id == m.recipient_id), None
            )
            if recip and recip.pra > 0.80:
                xm_done = any(
                    ma.donor_id == m.donor_id
                    and ma.recipient_id == m.recipient_id
                    and ma.crossmatch_pending
                    for ma in s.matches
                )
                if not xm_done:
                    dangerous += 1
        safety = max(0.0, 1.0 - dangerous * 0.5)

        # Organ waste penalty
        waste_penalty = s.organs_wasted * 0.15

        components = {
            "transplant_rate":  round(transplant_rate, 3),
            "quality":          round(quality, 3),
            "viability_margin": round(viability_margin, 3),
            "urgency_priority": round(urgency_priority, 3),
            "safety":           round(safety, 3),
        }
        aggregate = sum(
            components[k] * self.WEIGHTS[k] for k in components
        )
        aggregate = max(0.0, min(1.0, aggregate - waste_penalty))

        return {
            **components,
            "organs_wasted": s.organs_wasted,
            "aggregate": round(aggregate, 3),
        }
