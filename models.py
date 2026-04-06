"""
Transplant Logistics Environment — Typed Models
Pydantic v2 contracts for Action, Observation, State, and StepResult.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class OrganType(str, Enum):
    KIDNEY   = "kidney"
    LIVER    = "liver"
    HEART    = "heart"
    LUNG     = "lung"
    PANCREAS = "pancreas"

class BloodType(str, Enum):
    O_NEG  = "O-"
    O_POS  = "O+"
    A_NEG  = "A-"
    A_POS  = "A+"
    B_NEG  = "B-"
    B_POS  = "B+"
    AB_NEG = "AB-"
    AB_POS = "AB+"

class UrgencyTier(str, Enum):
    STATUS_1A = "1A"   # life support / ICU  (NHSBT: Super-urgent)
    STATUS_1B = "1B"   # hospitalised         (NHSBT: Urgent)
    STATUS_2  = "2"    # outpatient           (NHSBT: Routine)
    STATUS_7  = "7"    # temporarily inactive (NHSBT: Suspended)

class TransportMode(str, Enum):
    GROUND     = "ground"
    CHARTER    = "charter"
    COMMERCIAL = "commercial"

class ActionType(str, Enum):
    MATCH_ORGAN        = "match_organ"
    REJECT_MATCH       = "reject_match"
    DISPATCH_TRANSPORT = "dispatch_transport"
    REQUEST_CROSSMATCH = "request_crossmatch"
    NOTIFY_TEAM        = "notify_team"
    DECLINE_ORGAN      = "decline_organ"
    PASS_TO_NEXT       = "pass_to_next"


# ── Domain objects ─────────────────────────────────────────────────────────────

class Donor(BaseModel):
    donor_id:      str
    organ_type:    OrganType
    blood_type:    BloodType
    age:           int
    hospital_id:   str
    procurement_time_utc:      str            # ISO-8601
    viability_hours:           float          # hours organ remains viable
    hla_antigens:              List[str] = Field(default_factory=list)
    kdpi:                      Optional[float] = None   # kidney donor profile index 0–1
    cross_clamp_time_minutes:  int = 0        # cold ischaemia accumulation

class Recipient(BaseModel):
    recipient_id:  str
    organ_needed:  OrganType
    blood_type:    BloodType
    age:           int
    hospital_id:   str
    urgency:       UrgencyTier
    wait_days:     int
    hla_antibodies: List[str] = Field(default_factory=list)
    pra:           float = 0.0          # panel reactive antibody 0–1
    eGFR:          Optional[float] = None
    dialysis_days: Optional[int]   = None
    meld_score:    Optional[int]   = None    # liver (≈ UKELD in NHS context)
    hvpg:          Optional[float] = None

class Hospital(BaseModel):
    hospital_id:        str
    name:               str
    lat:                float
    lon:                float
    has_charter_access: bool = True
    # ── FIX: added nhs_trust field (used in environment.py & dispatch info) ──
    nhs_trust:          Optional[str] = None

class TransportLeg(BaseModel):
    from_hospital:    str
    to_hospital:      str
    mode:             TransportMode
    distance_km:      float
    duration_minutes: float

class MatchRecord(BaseModel):
    donor_id:                    str
    recipient_id:                str
    compatibility_score:         float    # 0–1
    transport_minutes:           float
    remaining_viability_minutes: float
    accepted:                    bool = False
    crossmatch_pending:          bool = False


# ── OpenEnv typed contracts ────────────────────────────────────────────────────

class TransplantAction(BaseModel):
    """Single agent action."""
    action_type:    ActionType
    donor_id:       Optional[str]           = None
    recipient_id:   Optional[str]           = None
    transport_mode: Optional[TransportMode] = None
    message:        Optional[str]           = None
    reason:         Optional[str]           = None

class TransplantObservation(BaseModel):
    """What the agent sees each step."""
    step:             int
    available_donors: List[Donor]
    waitlist:         List[Recipient]
    hospitals:        List[Hospital]
    pending_matches:  List[MatchRecord]
    elapsed_minutes:  float
    alerts:           List[str] = Field(default_factory=list)
    task_id:          str
    task_description: str

class TransplantState(BaseModel):
    """Full internal state (returned by state())."""
    task_id:                 str
    step:                    int
    donors:                  List[Donor]
    recipients:              List[Recipient]
    hospitals:               List[Hospital]
    matches:                 List[MatchRecord]
    elapsed_minutes:         float
    successful_transplants:  int
    organs_wasted:           int
    avg_wait_reduction_days: float
    action_log:              List[Dict[str, Any]] = Field(default_factory=list)

class StepResult(BaseModel):
    """Returned by step()."""
    observation: TransplantObservation
    reward:      float
    done:        bool
    info:        Dict[str, Any]
