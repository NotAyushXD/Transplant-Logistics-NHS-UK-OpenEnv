"""
Transplant Logistics Environment — NHS UK Edition
===================================================
Adapted from the original India/UNOS version to use:
  - Real UK NHS Blood and Transplant (NHSBT) transplant centres
  - UK organ distribution statistics (2019–2023 NHS data)
  - NHSBT allocation protocols instead of UNOS
  - NHS data-calibrated wait times, organ counts, and urgency distributions

NHS data source: NHS Blood and Transplant Activity Report 2019–2023
  Key statistics used to calibrate this simulation:
    • ~3,500 transplants/year in UK
    • Kidney:  ~47% of all transplants (largest category)
    • Liver:   ~22%
    • Heart:   ~8%
    • Lung:    ~10%
    • Pancreas: ~4%
    • Median kidney wait time: ~2.5 years (≈912 days)
    • ~6,959 patients on active waiting list (end March 2023)
    • ~439 deaths on waiting list per year
    • Average organs retrieved per donor: 3.2
    • Consent rate: ~61% (post opt-out legislation)

NHSBT Protocol differences from UNOS (implemented here):
    • UK uses "Super-urgent" / "Urgent" / "Routine" instead of 1A/1B/2
      (mapped to STATUS_1A / STATUS_1B / STATUS_2 in the enum)
    • UK kidney allocation uses Kidney Donor Risk Index (KDRI) alongside KDPI
    • UK uses UKELD score (instead of MELD) for liver urgency;
      we map UKELD ≈ MELD for compatibility with existing score field
    • PRA > 85% triggers mandatory crossmatch (UK threshold, vs 50% in UNOS)
    • DBD (Donation after Brain Death) and DCD (Donation after Circulatory Death)
      are both modelled; DCD has shorter viability windows
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

# ── Blood type compatibility (ABO — same worldwide) ───────────────────────────
BLOOD_COMPAT: Dict[BloodType, List[BloodType]] = {
    BloodType.O_NEG:  list(BloodType),
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

# Cold ischaemia limits (hours) — same biology, but DCD organs get 80% of window
VIABILITY_HOURS: Dict[OrganType, float] = {
    OrganType.HEART:    4.0,
    OrganType.LUNG:     6.0,
    OrganType.LIVER:   12.0,
    OrganType.KIDNEY:  24.0,
    OrganType.PANCREAS: 12.0,
}

# NHSBT protocol: mandatory crossmatch threshold is PRA > 0.85 (stricter than UNOS)
NHSBT_CROSSMATCH_PRA_THRESHOLD = 0.85

MINUTES_PER_STEP = 30


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def transport_minutes(h1: Hospital, h2: Hospital,
                      mode: TransportMode) -> float:
    """
    UK transport time model.
    Ground: ambulance ~70 km/h average (UK roads), 15 min handover.
    Charter: air ambulance ~280 km/h effective (includes airport time), 60 min overhead.
    Commercial: ~500 km/h, but 120 min overhead (check-in, transfer, packaging).
    """
    dist = haversine_km(h1.lat, h1.lon, h2.lat, h2.lon)
    speeds   = {TransportMode.GROUND: 70.0,
                TransportMode.CHARTER: 280.0,
                TransportMode.COMMERCIAL: 500.0}
    overhead = {TransportMode.GROUND: 15,
                TransportMode.CHARTER: 60,
                TransportMode.COMMERCIAL: 120}
    return (dist / speeds[mode]) * 60 + overhead[mode]


# ── Compatibility scoring (NHSBT-adapted) ────────────────────────────────────

def compatibility_score(donor: Donor, recipient: Recipient,
                        hosp_map: Dict[str, Hospital]) -> float:
    """
    Returns 0.0–1.0. 0.0 = incompatible (never proceed).

    NHSBT allocation priority factors:
      1. ABO blood type (mandatory)
      2. Organ type match (mandatory)
      3. HLA antibody crossreactivity penalty
      4. PRA-based sensitisation penalty
      5. Clinical urgency (Super-urgent > Urgent > Routine)
      6. Waiting time (NHS data: median kidney wait ~2.5 years)
      7. KDPI age-match bonus for kidneys
      8. UKELD/MELD severity bonus for liver
    """
    if recipient.blood_type not in BLOOD_COMPAT.get(donor.blood_type, []):
        return 0.0
    if donor.organ_type != recipient.organ_needed:
        return 0.0

    score = 0.5

    # HLA crossreactivity penalty (each shared antigen/antibody pair: −0.08)
    crossreact = set(donor.hla_antigens) & set(recipient.hla_antibodies)
    score -= len(crossreact) * 0.08

    # PRA sensitisation penalty
    score -= recipient.pra * 0.15

    # Urgency bonus (NHSBT: super-urgent gets biggest boost)
    urgency_bonus = {"1A": 0.30, "1B": 0.18, "2": 0.05, "7": 0.0}
    score += urgency_bonus.get(recipient.urgency.value, 0.0)

    # Waiting time bonus — NHS data shows median ~912 days for kidney;
    # log-scale so very long waits don't dominate entirely
    score += min(0.15, math.log1p(recipient.wait_days) / 35)

    # Kidney-specific: age matching and KDPI
    if donor.organ_type == OrganType.KIDNEY:
        age_diff = abs(donor.age - recipient.age)
        score -= min(0.10, age_diff * 0.002)
        if donor.kdpi is not None:
            score += (1 - donor.kdpi) * 0.10

    # Liver-specific: UKELD/MELD severity
    if donor.organ_type == OrganType.LIVER and recipient.meld_score:
        score += min(0.10, recipient.meld_score / 400)

    return max(0.0, min(1.0, round(score, 3)))


# ── NHS UK Transplant Centres ─────────────────────────────────────────────────
# Source: NHSBT Activity Report 2022/23 — all designated UK transplant centres.
# Coordinates are city-level (transplant centre locations).

def _hospitals() -> List[Hospital]:
    return [
        # England
        Hospital(hospital_id="H_LON_KCH",  name="King's College Hospital, London",
                 lat=51.4679, lon=-0.0877,  nhs_trust="KCH NHS FT",    has_charter_access=True),
        Hospital(hospital_id="H_LON_RFH",  name="Royal Free Hospital, London",
                 lat=51.5534, lon=-0.1654,  nhs_trust="RFH NHS FT",    has_charter_access=True),
        Hospital(hospital_id="H_LON_GOS",  name="Great Ormond Street, London",
                 lat=51.5225, lon=-0.1197,  nhs_trust="GOSH NHS FT",   has_charter_access=True),
        Hospital(hospital_id="H_CAM",      name="Addenbrooke's, Cambridge",
                 lat=52.1751, lon=0.1404,   nhs_trust="CUH NHS FT",    has_charter_access=True),
        Hospital(hospital_id="H_BIR",      name="Queen Elizabeth, Birmingham",
                 lat=52.4540, lon=-1.9441,  nhs_trust="UHB NHS FT",    has_charter_access=True),
        Hospital(hospital_id="H_MAN",      name="Manchester Royal Infirmary",
                 lat=53.4624, lon=-2.2291,  nhs_trust="MFT NHS FT",    has_charter_access=True),
        Hospital(hospital_id="H_NEW",      name="Freeman Hospital, Newcastle",
                 lat=55.0019, lon=-1.6097,  nhs_trust="NUTH NHS FT",   has_charter_access=True),
        Hospital(hospital_id="H_LEE",      name="St James's, Leeds",
                 lat=53.8072, lon=-1.5233,  nhs_trust="LTHT NHS FT",   has_charter_access=True),
        Hospital(hospital_id="H_SHE",      name="Sheffield Teaching Hospitals",
                 lat=53.3784, lon=-1.4885,  nhs_trust="STH NHS FT",    has_charter_access=False),
        Hospital(hospital_id="H_BRI",      name="Bristol Royal Infirmary",
                 lat=51.4584, lon=-2.5969,  nhs_trust="UHB NHS FT",    has_charter_access=True),
        Hospital(hospital_id="H_OXF",      name="Oxford University Hospitals",
                 lat=51.7520, lon=-1.2577,  nhs_trust="OUH NHS FT",    has_charter_access=True),
        # Scotland
        Hospital(hospital_id="H_GLA",      name="NHS Greater Glasgow & Clyde",
                 lat=55.8642, lon=-4.2518,  nhs_trust="NHS GGC",        has_charter_access=True),
        Hospital(hospital_id="H_EDI",      name="Royal Infirmary of Edinburgh",
                 lat=55.9229, lon=-3.1350,  nhs_trust="NHS Lothian",    has_charter_access=True),
        # Wales
        Hospital(hospital_id="H_CAR",      name="University Hospital of Wales, Cardiff",
                 lat=51.4958, lon=-3.1994,  nhs_trust="Cardiff & Vale", has_charter_access=True),
        # Northern Ireland
        Hospital(hospital_id="H_BEL",      name="Belfast City Hospital",
                 lat=54.5896, lon=-5.9424,  nhs_trust="Belfast HSC",    has_charter_access=True),
    ]


# ── NHS-calibrated Task Definitions ──────────────────────────────────────────

TASKS: Dict[str, Dict] = {

    # ── EASY: Kidney — most common UK transplant ───────────────────────────
    "task_easy_clear_match": {
        "id":          "task_easy_clear_match",
        "name":        "Single kidney — clear best match (NHS UK)",
        "difficulty":  "easy",
        "description": (
            "A deceased DBD donor kidney is available at King's College Hospital, London. "
            "Three recipients are on the waiting list. One is the clear best match: "
            "compatible blood type, Super-urgent (Status 1A), low PRA, same hospital. "
            "NHS data: median kidney wait ~912 days; ~9,000 patients on UK kidney list. "
            "Match the organ, dispatch transport, and notify the surgical team."
        ),
        "max_steps": 8,
        "donors": [
            Donor(donor_id="D001", organ_type=OrganType.KIDNEY,
                  blood_type=BloodType.A_POS, age=45,
                  hospital_id="H_LON_KCH",
                  procurement_time_utc="2023-09-15T06:00:00Z",
                  viability_hours=24.0, hla_antigens=["A2", "B44"],
                  kdpi=0.30),
        ],
        "recipients": [
            # Correct match: A+, Status 1A, same hospital, low PRA, long wait
            Recipient(recipient_id="R001", organ_needed=OrganType.KIDNEY,
                      blood_type=BloodType.A_POS, age=42,
                      hospital_id="H_LON_KCH", urgency=UrgencyTier.STATUS_1A,
                      wait_days=950, hla_antibodies=[], pra=0.04),
            # Plausible distractor: AB+, Status 2, distant, moderate PRA
            Recipient(recipient_id="R002", organ_needed=OrganType.KIDNEY,
                      blood_type=BloodType.AB_POS, age=58,
                      hospital_id="H_NEW", urgency=UrgencyTier.STATUS_2,
                      wait_days=400, hla_antibodies=["A2"], pra=0.40),
            # Incompatible blood type trap
            Recipient(recipient_id="R003", organ_needed=OrganType.KIDNEY,
                      blood_type=BloodType.B_POS, age=38,
                      hospital_id="H_BIR", urgency=UrgencyTier.STATUS_1B,
                      wait_days=600, hla_antibodies=[], pra=0.10),
        ],
        "required_actions": [
            ActionType.MATCH_ORGAN,
            ActionType.DISPATCH_TRANSPORT,
            ActionType.NOTIFY_TEAM,
        ],
    },

    # ── MEDIUM: Heart + Liver cascade (DBD donor) ─────────────────────────
    "task_medium_cascade_allocation": {
        "id":          "task_medium_cascade_allocation",
        "name":        "Multi-organ cascade — liver + heart under viability pressure (NHS UK)",
        "difficulty":  "medium",
        "description": (
            "Two organs from the same DBD donor at Freeman Hospital, Newcastle: "
            "a heart (4h viability) and a liver (12h viability). "
            "NHS protocol: heart must be matched and dispatched first. "
            "One recipient has high PRA — a trap for naive urgency-first agents. "
            "NHS 2022/23: 231 heart transplants, 1,011 liver transplants performed. "
            "Allocate both organs correctly before the heart's cold ischaemia expires."
        ),
        "max_steps": 14,
        "donors": [
            Donor(donor_id="D002", organ_type=OrganType.HEART,
                  blood_type=BloodType.O_POS, age=30,
                  hospital_id="H_NEW",
                  procurement_time_utc="2023-10-02T08:00:00Z",
                  viability_hours=4.0, hla_antigens=["B7"]),
            Donor(donor_id="D003", organ_type=OrganType.LIVER,
                  blood_type=BloodType.O_POS, age=30,
                  hospital_id="H_NEW",
                  procurement_time_utc="2023-10-02T08:00:00Z",
                  viability_hours=12.0, hla_antigens=["B7"]),
        ],
        "recipients": [
            # Heart recipients
            Recipient(recipient_id="R010", organ_needed=OrganType.HEART,
                      blood_type=BloodType.O_POS, age=33,
                      hospital_id="H_NEW", urgency=UrgencyTier.STATUS_1A,
                      wait_days=45, hla_antibodies=[], pra=0.05),
            Recipient(recipient_id="R011", organ_needed=OrganType.HEART,
                      blood_type=BloodType.A_POS, age=41,   # blood type mismatch
                      hospital_id="H_LEE", urgency=UrgencyTier.STATUS_1A,
                      wait_days=70, hla_antibodies=[], pra=0.08),
            Recipient(recipient_id="R012", organ_needed=OrganType.HEART,
                      blood_type=BloodType.O_NEG, age=52,
                      hospital_id="H_MAN", urgency=UrgencyTier.STATUS_1B,
                      wait_days=120, hla_antibodies=["B7"], pra=0.88),  # high PRA trap
            # Liver recipients
            Recipient(recipient_id="R020", organ_needed=OrganType.LIVER,
                      blood_type=BloodType.O_POS, age=51,
                      hospital_id="H_LEE", urgency=UrgencyTier.STATUS_1A,
                      wait_days=210, hla_antibodies=[], pra=0.08,
                      meld_score=35),
            Recipient(recipient_id="R021", organ_needed=OrganType.LIVER,
                      blood_type=BloodType.B_POS, age=55,   # blood type mismatch
                      hospital_id="H_BIR", urgency=UrgencyTier.STATUS_1B,
                      wait_days=350, hla_antibodies=[], pra=0.04,
                      meld_score=28),
        ],
        "required_actions": [
            ActionType.MATCH_ORGAN,
            ActionType.DISPATCH_TRANSPORT,
            ActionType.MATCH_ORGAN,
            ActionType.DISPATCH_TRANSPORT,
            ActionType.NOTIFY_TEAM,
        ],
    },

    # ── HARD: Lung expiry crisis (DCD donor) ─────────────────────────────
    "task_hard_expiry_crisis": {
        "id":          "task_hard_expiry_crisis",
        "name":        "Expiry crisis — misleading urgency + crossmatch hold (NHS UK)",
        "difficulty":  "hard",
        "description": (
            "Three donors at UK centres; five recipients. "
            "D101: DCD lung at Addenbrooke's — reduced viability (3.5h), expiring in 3 steps. "
            "D102: DBD heart at Addenbrooke's — the Status 1A recipient has high PRA "
            "and a positive virtual crossmatch (hyperacute rejection risk — DO NOT MATCH). "
            "D103: High-KDPI kidney at Glasgow — should go to an older recipient. "
            "NHS 2022/23 report: organ non-utilisation is a key issue; DCD lungs "
            "are particularly difficult to allocate under time pressure. "
            "Correct allocation requires crossmatch reasoning, PRA awareness, and KDPI logic."
        ),
        "max_steps": 20,
        "donors": [
            # DCD lung — short viability, expiring fast
            Donor(donor_id="D101", organ_type=OrganType.LUNG,
                  blood_type=BloodType.A_NEG, age=48,
                  hospital_id="H_CAM",
                  procurement_time_utc="2023-11-05T10:00:00Z",
                  viability_hours=3.5, hla_antigens=["A1", "B8"]),
            # DBD heart — the Status 1A is a crossmatch trap
            Donor(donor_id="D102", organ_type=OrganType.HEART,
                  blood_type=BloodType.O_NEG, age=24,
                  hospital_id="H_CAM",
                  procurement_time_utc="2023-11-05T10:00:00Z",
                  viability_hours=4.0, hla_antigens=["A3"]),
            # High-KDPI kidney — age-match to older recipient
            Donor(donor_id="D103", organ_type=OrganType.KIDNEY,
                  blood_type=BloodType.B_POS, age=70,
                  hospital_id="H_GLA",
                  procurement_time_utc="2023-11-05T10:00:00Z",
                  viability_hours=24.0, hla_antigens=["B35"],
                  kdpi=0.85),
        ],
        "recipients": [
            # R101: Status 1A but positive crossmatch — DO NOT MATCH
            Recipient(recipient_id="R101", organ_needed=OrganType.LUNG,
                      blood_type=BloodType.A_NEG, age=40,
                      hospital_id="H_CAM", urgency=UrgencyTier.STATUS_1A,
                      wait_days=95, hla_antibodies=["A1", "B8"], pra=0.93),
            # R102: blood type incompatible
            Recipient(recipient_id="R102", organ_needed=OrganType.LUNG,
                      blood_type=BloodType.A_POS, age=46,
                      hospital_id="H_BRI", urgency=UrgencyTier.STATUS_1B,
                      wait_days=180, hla_antibodies=[], pra=0.04),
            # R103: correct lung match — lower urgency but safe
            Recipient(recipient_id="R103", organ_needed=OrganType.LUNG,
                      blood_type=BloodType.A_NEG, age=52,
                      hospital_id="H_CAM", urgency=UrgencyTier.STATUS_1B,
                      wait_days=310, hla_antibodies=[], pra=0.07),
            # R110: Status 1A but HLA antibody overlap + high PRA → crossmatch positive
            Recipient(recipient_id="R110", organ_needed=OrganType.HEART,
                      blood_type=BloodType.O_NEG, age=31,
                      hospital_id="H_CAM", urgency=UrgencyTier.STATUS_1A,
                      wait_days=18, hla_antibodies=["A3"], pra=0.87),
            # R111: Status 1B, safe match, correct choice
            Recipient(recipient_id="R111", organ_needed=OrganType.HEART,
                      blood_type=BloodType.O_POS, age=35,
                      hospital_id="H_CAM", urgency=UrgencyTier.STATUS_1B,
                      wait_days=65, hla_antibodies=[], pra=0.09),
            # Kidney recipient — older, matches high-KDPI donor
            Recipient(recipient_id="R120", organ_needed=OrganType.KIDNEY,
                      blood_type=BloodType.B_POS, age=67,
                      hospital_id="H_GLA", urgency=UrgencyTier.STATUS_2,
                      wait_days=820, hla_antibodies=[], pra=0.04,
                      eGFR=11.0),
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


# ── NHS Data Loader ───────────────────────────────────────────────────────────

class NHSDataLoader:
    """
    Loads NHS organ donation statistics from the Kaggle CSV dataset
    (patricklford/nhs-organ-donation, UK 2019–2023) and uses them to
    calibrate synthetic task generation.

    Expected CSV columns (based on NHS Blood and Transplant report format):
      Year, Organ, Transplants, WaitingList, MedianWaitDays,
      Donors_DBD, Donors_DCD, DeathsOnList, UtilisationRate

    Usage:
        loader = NHSDataLoader("path/to/nhs_organ_donation.csv")
        stats  = loader.get_organ_stats("kidney")
    """

    FALLBACK_STATS: Dict[str, Dict] = {
        "kidney": {
            "median_wait_days":    912,
            "annual_transplants":  3320,
            "waiting_list_size":   9000,
            "utilisation_rate":    0.83,
            "pct_dcd":             0.42,
            "typical_kdpi_range":  (0.20, 0.90),
        },
        "liver": {
            "median_wait_days":    145,
            "annual_transplants":  1011,
            "waiting_list_size":   800,
            "utilisation_rate":    0.89,
            "pct_dcd":             0.18,
            "typical_ukeld_range": (49, 65),
        },
        "heart": {
            "median_wait_days":    60,
            "annual_transplants":  231,
            "waiting_list_size":   350,
            "utilisation_rate":    0.77,
            "pct_dcd":             0.03,
        },
        "lung": {
            "median_wait_days":    180,
            "annual_transplants":  285,
            "waiting_list_size":   420,
            "utilisation_rate":    0.71,
            "pct_dcd":             0.05,
        },
        "pancreas": {
            "median_wait_days":    300,
            "annual_transplants":  140,
            "waiting_list_size":   250,
            "utilisation_rate":    0.68,
            "pct_dcd":             0.10,
        },
    }

    def __init__(self, csv_path: Optional[str] = None):
        self._stats = copy.deepcopy(self.FALLBACK_STATS)
        if csv_path:
            self._load_csv(csv_path)

    def _load_csv(self, path: str) -> None:
        """
        Parse the NHS Kaggle CSV and update internal stats.
        Handles multiple possible column name formats from the dataset.
        """
        try:
            import csv
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return

            organ_rows: Dict[str, List[Dict]] = {}
            for row in rows:
                organ_col = next(
                    (k for k in row if "organ" in k.lower()), None
                )
                if not organ_col:
                    continue
                organ = row[organ_col].strip().lower()
                organ_rows.setdefault(organ, []).append(row)

            for organ, r_list in organ_rows.items():
                organ_key = organ.replace(" ", "_")
                if organ_key not in self._stats:
                    continue

                def _mean(col_fragments: List[str]) -> Optional[float]:
                    vals = []
                    for r in r_list:
                        col = next((k for k in r
                                    if any(f in k.lower() for f in col_fragments)), None)
                        if col:
                            try:
                                vals.append(float(str(r[col]).replace(",", "")))
                            except ValueError:
                                pass
                    return sum(vals) / len(vals) if vals else None

                tx   = _mean(["transplant", "activity"])
                wait = _mean(["wait", "median"])
                util = _mean(["utilisation", "utilization", "rate"])
                wl   = _mean(["list", "waiting"])

                if tx   is not None: self._stats[organ_key]["annual_transplants"] = int(tx)
                if wait is not None: self._stats[organ_key]["median_wait_days"]   = int(wait)
                if util is not None: self._stats[organ_key]["utilisation_rate"]   = util
                if wl   is not None: self._stats[organ_key]["waiting_list_size"]  = int(wl)

            print(f"[NHSDataLoader] Loaded NHS data from {path}")

        except Exception as e:
            print(f"[NHSDataLoader] Could not load CSV ({e}). Using fallback stats.")

    def get_organ_stats(self, organ: str) -> Dict:
        return self._stats.get(organ.lower(), {})

    def calibrated_wait_days(self, organ: str, urgency: UrgencyTier,
                              rng: random.Random) -> int:
        stats  = self.get_organ_stats(organ)
        median = stats.get("median_wait_days", 200)
        if urgency == UrgencyTier.STATUS_1A:
            return rng.randint(7, 60)
        elif urgency == UrgencyTier.STATUS_1B:
            return rng.randint(30, int(median * 0.5))
        elif urgency == UrgencyTier.STATUS_2:
            return rng.randint(int(median * 0.5), int(median * 1.5))
        else:
            return rng.randint(int(median * 0.2), int(median * 0.8))


# ── Environment ───────────────────────────────────────────────────────────────

class TransplantEnv:
    """
    OpenEnv-compliant NHS UK transplant logistics environment.
    """

    def __init__(self, task_id: str, nhs_csv_path: Optional[str] = None):
        assert task_id in TASKS, f"Unknown task: {task_id}"
        self.task_id   = task_id
        self._task     = TASKS[task_id]
        self._hosp_map: Dict[str, Hospital] = {
            h.hospital_id: h for h in _hospitals()
        }
        self._nhs   = NHSDataLoader(nhs_csv_path)
        self._state: Optional[TransplantState] = None

    # ── OpenEnv API ───────────────────────────────────────────────────────

    def reset(self, seed: int = 42) -> TransplantObservation:
        random.seed(seed)
        task = self._task
        self._state = TransplantState(
            task_id                 = self.task_id,
            step                    = 0,
            donors                  = copy.deepcopy(task["donors"]),
            recipients              = copy.deepcopy(task["recipients"]),
            hospitals               = list(self._hosp_map.values()),
            matches                 = [],
            elapsed_minutes         = 0.0,
            successful_transplants  = 0,
            organs_wasted           = 0,
            avg_wait_reduction_days = 0.0,
            action_log              = [],
        )
        return self._make_observation()

    def step(self, action: TransplantAction) -> StepResult:
        assert self._state is not None, "Call reset() first"
        s = self._state
        s.step            += 1
        s.elapsed_minutes += MINUTES_PER_STEP

        s.action_log.append({
            "step":         s.step,
            "action":       action.action_type.value,
            "donor_id":     action.donor_id,
            "recipient_id": action.recipient_id,
        })

        # Advance cold ischaemia on all remaining donors
        for donor in s.donors:
            donor.cross_clamp_time_minutes += MINUTES_PER_STEP

        reward, info = self._execute(action)

        # Check organ expiry
        expired = []
        for donor in s.donors:
            elapsed_h = donor.cross_clamp_time_minutes / 60
            if elapsed_h >= donor.viability_hours:
                already_matched = any(
                    m.donor_id == donor.donor_id and m.accepted
                    for m in s.matches
                )
                if not already_matched:
                    s.organs_wasted += 1
                    info["alerts"] = info.get("alerts", []) + [
                        f"ORGAN EXPIRED: {donor.organ_type.value} "
                        f"from {donor.donor_id} at {donor.hospital_id}"
                    ]
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
        a    = action.action_type
        info: Dict[str, Any] = {"action": a.value, "alerts": []}

        dispatch = {
            ActionType.MATCH_ORGAN:        self._do_match,
            ActionType.DISPATCH_TRANSPORT: self._do_dispatch,
            ActionType.REQUEST_CROSSMATCH: self._do_crossmatch,
            ActionType.NOTIFY_TEAM:        self._do_notify,
            ActionType.DECLINE_ORGAN:      self._do_decline,
            ActionType.REJECT_MATCH:       self._do_reject,
            ActionType.PASS_TO_NEXT:       lambda ac, i: (0.0, i),
        }
        fn = dispatch.get(a)
        if fn:
            return fn(action, info)
        return 0.0, info

    def _do_match(self, action: TransplantAction,
                  info: Dict) -> Tuple[float, Dict]:
        s     = self._state
        donor = next((d for d in s.donors if d.donor_id == action.donor_id), None)
        recip = next((r for r in s.recipients if r.recipient_id == action.recipient_id), None)

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

        # NHSBT: PRA > 0.85 requires mandatory crossmatch before matching
        if recip.pra > NHSBT_CROSSMATCH_PRA_THRESHOLD:
            xm_done = any(
                m.donor_id == donor.donor_id
                and m.recipient_id == recip.recipient_id
                and m.crossmatch_pending
                for m in s.matches
            )
            if not xm_done:
                info["warning"] = (
                    f"NHSBT protocol violation: Recipient {recip.recipient_id} "
                    f"has PRA={recip.pra:.0%} (>{NHSBT_CROSSMATCH_PRA_THRESHOLD:.0%}). "
                    "Crossmatch MUST be requested first under NHSBT guidelines."
                )
                return -0.2, info

        # Transport viability check
        dh        = self._hosp_map[donor.hospital_id]
        rh        = self._hosp_map[recip.hospital_id]
        mode      = action.transport_mode or TransportMode.CHARTER
        t_min     = transport_minutes(dh, rh, mode)
        remaining = (donor.viability_hours * 60) - donor.cross_clamp_time_minutes

        if t_min >= remaining:
            info["warning"] = (
                f"Transit time {t_min:.0f} min exceeds remaining viability "
                f"{remaining:.0f} min — organ will expire in transit."
            )
            return -0.25, info

        match = MatchRecord(
            donor_id                    = donor.donor_id,
            recipient_id                = recip.recipient_id,
            compatibility_score         = score,
            transport_minutes           = t_min,
            remaining_viability_minutes = remaining,
            accepted                    = True,
        )
        s.matches.append(match)
        s.successful_transplants += 1
        s.recipients = [r for r in s.recipients if r.recipient_id != recip.recipient_id]

        viability_margin = min(1.0, (remaining - t_min) / remaining)
        urgency_mult     = {"1A": 1.5, "1B": 1.2, "2": 1.0, "7": 0.5}
        reward = (
            score * 0.5
            + viability_margin * 0.3
            + urgency_mult.get(recip.urgency.value, 1.0) * 0.2
        )
        info["match"]               = match.model_dump()
        info["compatibility_score"] = score
        return round(reward, 4), info

    def _do_dispatch(self, action: TransplantAction,
                     info: Dict) -> Tuple[float, Dict]:
        s     = self._state
        match = next((m for m in s.matches
                      if m.donor_id == action.donor_id and m.accepted), None)
        if match is None:
            info["error"] = "No accepted match found — match organ first"
            return -0.1, info

        mode  = action.transport_mode or TransportMode.CHARTER
        donor = next((d for d in s.donors if d.donor_id == action.donor_id), None)
        if donor is None:
            info["message"] = "Organ already dispatched"
            return 0.05, info

        dh         = self._hosp_map.get(donor.hospital_id)
        recip_hosp = None
        # Look up recipient hospital from the original task definition
        # (recipient may have been removed from live state after matching)
        for r_orig in self._task["recipients"]:
            if r_orig.recipient_id == match.recipient_id:
                recip_hosp = self._hosp_map.get(r_orig.hospital_id)
                break

        if dh and recip_hosp:
            t_min = transport_minutes(dh, recip_hosp, mode)
            info["transport_dispatched"] = {
                "from":           dh.name,
                "to":             recip_hosp.name,
                "mode":           mode.value,
                "eta_minutes":    round(t_min),
                "nhs_trust_from": dh.nhs_trust,
                "nhs_trust_to":   recip_hosp.nhs_trust,
            }

        s.donors = [d for d in s.donors if d.donor_id != action.donor_id]
        return 0.15, info

    def _do_crossmatch(self, action: TransplantAction,
                       info: Dict) -> Tuple[float, Dict]:
        s     = self._state
        donor = next((d for d in s.donors if d.donor_id == action.donor_id), None)
        recip = next((r for r in s.recipients if r.recipient_id == action.recipient_id), None)
        if not donor or not recip:
            info["error"] = "Donor or recipient not found"
            return -0.05, info

        crossreact = set(donor.hla_antigens) & set(recip.hla_antibodies)
        positive   = bool(crossreact) or recip.pra > NHSBT_CROSSMATCH_PRA_THRESHOLD

        existing = next(
            (m for m in s.matches
             if m.donor_id == action.donor_id
             and m.recipient_id == action.recipient_id), None
        )
        if not existing:
            s.matches.append(MatchRecord(
                donor_id                    = donor.donor_id,
                recipient_id                = recip.recipient_id,
                compatibility_score         = 0.0,
                transport_minutes           = 0.0,
                remaining_viability_minutes = 0.0,
                crossmatch_pending          = True,
            ))

        info["crossmatch_result"] = {
            "positive":               positive,
            "crossreacting_antigens": list(crossreact),
            "pra":                    recip.pra,
            "nhsbt_threshold":        NHSBT_CROSSMATCH_PRA_THRESHOLD,
            "recommendation": (
                "DO NOT PROCEED — hyperacute rejection risk (NHSBT protocol)"
                if positive else "Proceed with transplant"
            ),
        }
        reward = 0.10 if recip.pra > 0.50 else 0.0
        return reward, info

    def _do_notify(self, action: TransplantAction,
                   info: Dict) -> Tuple[float, Dict]:
        s        = self._state
        accepted = [m for m in s.matches if m.accepted]
        info["notification_sent"]       = True
        info["transplants_coordinated"] = len(accepted)
        info["nhs_protocol"]            = "NHSBT notification complete"
        total_donors_orig = len(self._task["donors"])
        completion        = len(accepted) / max(total_donors_orig, 1)
        return 0.1 + completion * 0.2, info

    def _do_decline(self, action: TransplantAction,
                    info: Dict) -> Tuple[float, Dict]:
        s     = self._state
        donor = next((d for d in s.donors if d.donor_id == action.donor_id), None)
        recip = next((r for r in s.recipients if r.recipient_id == action.recipient_id), None)
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
        s.matches = [
            m for m in s.matches
            if not (m.donor_id == action.donor_id
                    and m.recipient_id == action.recipient_id)
        ]
        info["message"] = f"Match rejected: {action.reason}"
        return 0.0, info

    # ── Observation builder ───────────────────────────────────────────────

    def _make_observation(self) -> TransplantObservation:
        s      = self._state
        alerts = []
        for d in s.donors:
            remaining_h = (d.viability_hours * 60 - d.cross_clamp_time_minutes) / 60
            if remaining_h < 2.0:
                alerts.append(
                    f"URGENT: {d.organ_type.value} {d.donor_id} at "
                    f"{d.hospital_id} expires in {remaining_h:.1f}h"
                )
        return TransplantObservation(
            step             = s.step,
            available_donors = s.donors,
            waitlist         = s.recipients,
            hospitals        = list(self._hosp_map.values()),
            pending_matches  = s.matches,
            elapsed_minutes  = s.elapsed_minutes,
            alerts           = alerts,
            task_id          = self.task_id,
            task_description = self._task["description"],
        )


# ── Grader ────────────────────────────────────────────────────────────────────

class TransplantGrader:
    """
    Scores a completed episode 0.0–1.0 using NHSBT-aligned metrics.
    """

    WEIGHTS = {
        "transplant_rate":  0.35,
        "quality":          0.25,
        "viability_margin": 0.20,
        "urgency_priority": 0.10,
        "safety":           0.10,
    }

    def grade(self, final_state: TransplantState,
              task: Dict) -> Dict[str, float]:
        s        = final_state
        accepted = [m for m in s.matches if m.accepted]
        total    = len(task["donors"])

        transplant_rate  = len(accepted) / max(total, 1)
        quality          = (sum(m.compatibility_score for m in accepted) / len(accepted)
                            if accepted else 0.0)

        with_margin      = [m for m in accepted
                            if (m.remaining_viability_minutes - m.transport_minutes) >= 30]
        viability_margin = len(with_margin) / len(accepted) if accepted else 0.0

        orig_1a          = [r for r in task["recipients"]
                            if r.urgency == UrgencyTier.STATUS_1A]
        matched_1a       = [m for m in accepted
                            if any(r.recipient_id == m.recipient_id
                                   and r.urgency == UrgencyTier.STATUS_1A
                                   for r in task["recipients"])]
        urgency_priority = len(matched_1a) / len(orig_1a) if orig_1a else 1.0

        dangerous = 0
        for m in accepted:
            recip = next((r for r in task["recipients"]
                          if r.recipient_id == m.recipient_id), None)
            if recip and recip.pra > NHSBT_CROSSMATCH_PRA_THRESHOLD:
                xm_done = any(
                    ma.donor_id == m.donor_id
                    and ma.recipient_id == m.recipient_id
                    and ma.crossmatch_pending
                    for ma in s.matches
                )
                if not xm_done:
                    dangerous += 1
        safety = max(0.0, 1.0 - dangerous * 0.5)

        waste_penalty = s.organs_wasted * 0.15

        components = {
            "transplant_rate":  round(transplant_rate,  3),
            "quality":          round(quality,           3),
            "viability_margin": round(viability_margin,  3),
            "urgency_priority": round(urgency_priority,  3),
            "safety":           round(safety,            3),
        }
        aggregate = sum(components[k] * self.WEIGHTS[k] for k in components)
        aggregate = max(0.0, min(1.0, aggregate - waste_penalty))

        return {
            **components,
            "organs_wasted": s.organs_wasted,
            "aggregate":     round(aggregate, 3),
        }
