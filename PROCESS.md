# Transplant Logistics NHS — Solution Process Document

**Project:** Transplant Logistics OpenEnv — NHS UK Edition  
**Author:** Ayush Pant  
**Date:** April 2026  
**Version:** 1.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [System Design](#4-system-design)
5. [Implementation](#5-implementation)
6. [Challenges Encountered and How We Solved Them](#6-challenges-encountered-and-how-we-solved-them)
7. [Results and Evaluation](#7-results-and-evaluation)
8. [Known Limitations](#8-known-limitations)
9. [Future Work](#9-future-work)
10. [Conclusion](#10-conclusion)

---

## 1. Executive Summary

This project builds a Reinforcement Learning (RL) environment that simulates organ transplant allocation decisions for the NHS. A Large Language Model (LLM) acts as an AI agent, receiving donor and recipient data and deciding how to allocate organs before their viability windows close — mirroring the real decisions made by national transplant coordinators at NHS Blood and Transplant (NHSBT).

The system is calibrated to real NHS 2022/23 statistics, enforces actual NHSBT clinical protocols, and uses UK transplant centre locations for transport modelling. An evaluation framework scores the agent across five clinically meaningful metrics. The final pipeline achieves a mean aggregate score of **0.917** across three difficulty-graded tasks, compared to a **0.540** GPT-4o-mini baseline — a **+0.38 improvement** using a free inference backend.

---

## 2. Problem Statement

### 2.1 Clinical Context

Organ transplant allocation is one of the most time-critical and high-stakes decision-making processes in healthcare. In the UK alone:

- **~6,959 patients** were on the active transplant waiting list at end March 2023
- **~439 patients die each year** while waiting for a suitable organ
- **~3,500 transplants** are performed annually by NHS Blood and Transplant
- Every allocation decision must balance **biological compatibility**, **clinical urgency**, **organ viability**, and **transport logistics** — simultaneously, under extreme time pressure

The decisions are extraordinarily complex. A transplant coordinator must consider:
- **ABO blood type compatibility** between donor and recipient
- **HLA antigen and antibody matching** to assess rejection risk
- **Panel Reactive Antibody (PRA)** scores — highly sensitised patients require mandatory crossmatch testing before any match can proceed
- **Cold ischaemia time** — the window during which an organ remains viable outside the body (as short as 4 hours for a heart)
- **Transport logistics** — road ambulances, air charter, and commercial flights each have different speed and overhead profiles
- **Clinical urgency** — Status 1A (life support) patients take priority over Status 1B and routine patients
- **KDPI and MELD scores** — additional organ and recipient quality metrics used for kidney and liver allocation

Human coordinators manage this complexity through years of training and established protocols. The question this project addresses is: **can an AI agent be trained to replicate and potentially surpass human-level allocation decisions in a simulated environment?**

### 2.2 The AI Problem

Training an AI system to make organ allocation decisions poses several challenges that standard supervised learning approaches cannot easily address:

- **Sequential decision-making** — a coordinator does not make one decision; they take a sequence of actions (crossmatch, match, dispatch, notify) where each action changes the state of the world and affects what decisions remain available
- **Dense reward signals** — the quality of an allocation decision cannot be judged in isolation; it depends on what else was done in the same episode (did the organ arrive in time? was the right crossmatch requested first?)
- **Safety constraints** — certain actions (matching a high-PRA recipient without crossmatch, attempting a blood-type-incompatible match) carry real medical risk and must be penalised, not just avoided
- **Time pressure** — organs expire if not matched and dispatched within their cold ischaemia window, requiring the agent to prioritise correctly under a ticking clock
- **Sparse ground truth** — there is no labelled dataset of "correct" allocation decisions; correctness is defined by clinical protocols and outcome metrics

### 2.3 Scope

This project builds:
1. A **simulation environment** implementing NHSBT allocation protocols and NHS UK hospital infrastructure
2. An **LLM-based agent** that reasons over structured clinical observations to produce allocation decisions
3. An **evaluation framework** that grades agent performance across five clinically meaningful metrics
4. A **fine-tuning pipeline** using GRPO to improve agent performance through reinforcement learning

---

## 3. Solution Overview

### 3.1 Approach

We model organ transplant allocation as a **sequential decision-making problem** and implement it as an RL environment following the OpenEnv v1 framework. Rather than training a traditional policy network from scratch — which would require enormous amounts of clinical data — we use a **Large Language Model as the agent**, leveraging its pre-trained knowledge of medical concepts, clinical protocols, and logical reasoning.

The agent receives a structured natural language observation at each step, reasons about the situation, and outputs a JSON action. The environment enforces NHSBT clinical rules, computes a reward signal, and advances the simulation state.

This approach has several advantages over traditional RL:
- The LLM's pre-trained knowledge of medicine and logistics provides a strong starting point without domain-specific training data
- The agent can explain its reasoning (unlike a policy network whose decisions are opaque)
- Post-training via GRPO can further improve performance without requiring labelled allocation decisions

### 3.2 High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        RL EPISODE LOOP                         │
│                                                                │
│  ┌─────────────┐    obs_to_prompt()    ┌──────────────────┐   │
│  │  Environment│ ──────────────────── ▶│   LLM Agent      │   │
│  │             │                       │                  │   │
│  │ - 15 NHS    │ ◀─────────────────── │ - Groq / HF /    │   │
│  │   hospitals │    parse_action()     │   Local model    │   │
│  │ - NHSBT     │                       │ - Multi-turn     │   │
│  │   protocols │    step(action)       │   history        │   │
│  │ - Reward fn │ ──────────────────── ▶│                  │   │
│  └─────────────┘                       └──────────────────┘   │
│         │                                                      │
│         ▼                                                      │
│  ┌─────────────┐                                               │
│  │   Grader    │  0.0–1.0 aggregate score across 5 metrics    │
│  └─────────────┘                                               │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. System Design

### 4.1 Data Foundation

The environment is calibrated using two data sources:

**NHS Blood and Transplant Activity Report 2022/23**  
Used to set fallback statistics for all five organ types:

| Organ | Median wait | Annual transplants | Utilisation rate |
|-------|------------|-------------------|-----------------|
| Kidney | 912 days | 3,320 | 83% |
| Liver | 145 days | 1,011 | 89% |
| Heart | 60 days | 231 | 77% |
| Lung | 180 days | 285 | 71% |
| Pancreas | 300 days | 140 | 68% |

**Kaggle NHS Organ Donation Dataset** (`patricklford/nhs-organ-donation`)  
When provided, this CSV updates the fallback statistics with actual reported figures. The `NHSDataLoader` class handles parsing with column name normalisation to handle format variations.

### 4.2 Environment Design

**State representation (`TransplantState`)**  
The full environment state holds donors, recipients, hospitals, match records, elapsed time, and outcome counters. All objects are typed with Pydantic v2 models, enforcing correctness at every system boundary.

**Step mechanics**  
Each step advances cold ischaemia by 30 simulated minutes on every active donor. Organs that exceed their viability limit are removed and counted as wasted, incurring a 0.15 penalty per organ in the final grade.

**Compatibility scoring**  
A deterministic function returns 0.0 for illegal matches (blood type incompatibility, organ type mismatch) and a 0.0–1.0 score otherwise, incorporating HLA crossreactivity, PRA level, urgency tier, waiting time, KDPI, and MELD/UKELD.

**Transport model**  
Travel times use the Haversine formula on real UK hospital coordinates (city-level). Three modes are available — ground ambulance, air charter, and commercial flight — each with calibrated speeds and handling overhead.

**NHSBT protocol enforcement**  
- PRA > 85% triggers a mandatory crossmatch requirement before any match can proceed (stricter than the UNOS threshold of 50%)
- Positive crossmatch (HLA antigen/antibody overlap) prevents matching
- Blood type incompatibility returns an immediate −0.30 reward

### 4.3 Task Design

Three fixed scenarios are designed to test progressively harder reasoning capabilities:

| Task | Organs | Max steps | Key challenge |
|------|--------|-----------|---------------|
| Easy | 1 kidney | 8 | Basic blood type + urgency reasoning |
| Medium | 1 heart + 1 liver | 14 | Time ordering + high-PRA trap |
| Hard | 1 lung + 1 heart + 1 kidney | 20 | Crossmatch protocol + expiry crisis + KDPI logic |

Each task contains deliberately placed traps — recipients with high PRA, incompatible blood types, or misleadingly high urgency — to test whether the agent follows clinical protocol rather than naive heuristics.

### 4.4 Evaluation Framework

The `TransplantGrader` produces a 0.0–1.0 aggregate from five weighted components:

| Component | Weight | Description |
|-----------|--------|-------------|
| `transplant_rate` | 35% | Fraction of available organs matched |
| `quality` | 25% | Mean compatibility score of accepted matches |
| `viability_margin` | 20% | Fraction of transplants with ≥30 min buffer remaining |
| `urgency_priority` | 10% | Fraction of Status 1A patients matched |
| `safety` | 10% | No high-PRA matches without prior crossmatch |

---

## 5. Implementation

### 5.1 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Environment | Python + Pydantic v2 | Type-safe state management, clear contracts |
| API server | FastAPI | Async HTTP, automatic OpenAPI docs, OpenEnv compliant |
| LLM inference | Groq API / HF / local | Groq provides free, fast inference; local for offline use |
| Fine-tuning | TRL GRPOTrainer | Efficient RL post-training without a critic network |
| Data modelling | HuggingFace datasets | Standard format for GRPO training data |

### 5.2 Agent Design

The agent is a **prompt-based LLM wrapper** that converts environment observations into structured JSON actions.

**System prompt**  
Encodes the strict NHSBT workflow order (match → dispatch → repeat → notify), blood type compatibility rules, PRA thresholds, cold ischaemia limits, KDPI guidance, and JSON output format. The prompt treats the workflow order as non-negotiable, which was critical for preventing the agent from skipping dispatch steps.

**Observation prompt (`obs_to_prompt`)**  
The key design decisions in the observation prompt are:

- **Workflow state banner** — tells the model exactly what action is required next based on current state (if a match is accepted but not dispatched, the banner says "NEXT ACTION REQUIRED: dispatch_transport" explicitly)
- **Recipient grouping per donor** — compatible recipients are listed under their corresponding donor, pre-filtered by blood type and organ type, preventing the model from attempting illegal cross-organ matches
- **Transport feasibility flags** — for each compatible recipient, the prompt shows the fastest reachable transport mode and estimated ETA, with a ✓/✗ flag indicating whether the organ can physically reach them within the viability window
- **Crossmatch clearance tracking** — recipients with completed crossmatch requests are marked inline
- **Incompatible recipients section** — listed separately with a DO NOT ATTEMPT label

**Multi-turn history**  
The full conversation history is passed to the model on each step, allowing it to reason about what it has already done. History is capped at the last 4 exchanges to manage token usage.

**Action parsing**  
The `parse_action` function extracts JSON from raw model output, handling markdown code fences that models sometimes add. Falls back to `pass_to_next` on parse failure (0 reward, no penalty).

### 5.3 GRPO Fine-Tuning

The training script implements a three-phase loop:

1. **Warm-start rollouts** — run the current model through all tasks to collect (prompt, response, reward) triples
2. **GRPO update** — for each prompt, multiple responses are sampled; responses that scored above the group average are reinforced, those below are penalised; a KL penalty (β=0.04) prevents the policy from drifting too far from the base model
3. **Post-training evaluation** — the updated model is evaluated on all tasks to measure improvement

---

## 6. Challenges Encountered and How We Solved Them

### 6.1 Critical Bug: Pydantic Crash on Startup

**Problem**  
The environment crashed immediately on startup with a `ValidationError`. Every `Hospital` object in `environment.py` was constructed with a `nhs_trust` keyword argument, but the `Hospital` Pydantic model in `models.py` did not define that field. In Pydantic v2, extra fields passed to `__init__` raise a `ValidationError` — there is no silent discard.

**Impact**  
The server would not start. No task could run.

**Solution**  
Added `nhs_trust: Optional[str] = None` to the `Hospital` model in `models.py`. The field is used downstream in `_do_dispatch` to populate the transport dispatch info with NHS trust names.

---

### 6.2 Wrong File Names and Locations

**Problem**  
`inference.py` was at the project root but referenced throughout the codebase as `baseline/inference_hf.py`. Similarly, `train_grpo.py` was at the root but referenced as `training/train_grpo.py`. Python imports from other modules assumed the correct paths.

**Solution**  
Moved `inference.py` → `baseline/inference_hf.py` and `train_grpo.py` → `training/train_grpo.py`. Created `__init__.py` files in all three package directories (`server/`, `baseline/`, `training/`) so Python treats them as importable packages.

---

### 6.3 Missing `nhs_data_explorer.py`

**Problem**  
The run guide's Step 3 referenced `python nhs_data_explorer.py` as the smoke test, but the file did not exist in the codebase.

**Solution**  
Created `nhs_data_explorer.py` from scratch with two functions: a statistics printer that displays NHS data by organ type, and a heuristic agent that runs through all three tasks to verify the full pipeline is wired up before running the LLM agent.

---

### 6.4 HuggingFace 402 Payment Required

**Problem**  
The original codebase used the HuggingFace Inference API as its only cloud backend. During the first real run, the API returned a `402 Payment Required` error because `Qwen2.5-1.5B-Instruct` had been moved to the `featherless-ai` paid provider tier. The free HF credits were exhausted.

**Impact**  
The agent stopped mid-run on the hard task, leaving the episode incomplete and producing no final score.

**Solution**  
Added **Groq** as a new inference backend. Groq provides a generous free tier (~14,400 requests/day), speeds of ~1 second per step, and supports models with higher token limits than the HF free tier. The default backend was changed from `api` to `groq`, and the default model set to `llama-3.3-70b-versatile`.

---

### 6.5 Agent Looping on Wrong Actions (Medium Task)

**Problem**  
On the medium task, the agent repeatedly tried to match the liver donor (D003) to heart recipients (R012, R011), generating a sequence of `−0.30` incompatible match penalties. The heart (D002) expired because it was never dispatched. The agent scored −1.68 total reward.

**Root cause**  
The original observation prompt listed all recipients in a flat list without filtering by organ type or blood type. The model had no clear signal about which recipients were compatible with which donors, and no instruction about what action came next after a successful match.

**Solution**  
Rewrote `obs_to_prompt` with two key changes:
1. **Recipient grouping per donor** — compatible recipients (pre-filtered by blood type and organ type) are listed directly under their corresponding donor
2. **Workflow state banner** — a prominent "NEXT REQUIRED ACTION" line at the top of every prompt tells the model whether it should be matching, dispatching, or notifying based on current state

Also rewrote the system prompt to make the match → dispatch → repeat → notify order an explicit, non-negotiable rule.

---

### 6.6 413 Token Limit Error (Hard Task)

**Problem**  
On the hard task, the Groq API returned a `413 Request Too Large` error at step 8. The full conversation history (8 user observations × ~600 tokens + 7 assistant replies + system prompt) exceeded `llama-3.1-8b-instant`'s 6,000 tokens-per-minute limit.

**Root cause**  
The conversation history grew linearly with steps. The hard task has up to 20 steps, making the context window overflow inevitable with small models.

**Solution**  
Two changes:
1. **History truncation** — `max_history_turns=4` keeps only the last 4 exchanges in the message history. The current state is always fully described in the latest observation, so dropping older turns does not lose critical information
2. **Model upgrade** — changed the default Groq model from `llama-3.1-8b-instant` (6,000 TPM) to `llama-3.3-70b-versatile` (12,000 TPM), which handles the full hard task history without truncation while also providing substantially better reasoning quality
3. **Auto-retry** — if a 413 still occurs, the code automatically retries with the last 2 exchanges only before failing

---

### 6.7 Agent Attempting Unreachable Transport Routes (Hard Task)

**Problem**  
On step 1 of the hard task, the agent tried to match D101 (lung, A−, at Addenbrooke's Cambridge) to R102 (at Bristol) via ground transport. Ground transit time was 190 minutes; the lung only had 180 minutes of viability remaining. The match was rejected by the environment. The agent then failed to recover, eventually causing the lung to expire.

**Root cause**  
The observation prompt listed compatible recipients but gave no indication of whether the organ could physically reach them in time. The agent had to guess transport feasibility with no data.

**Solution**  
Added transport feasibility calculation to `obs_to_prompt`. For each compatible recipient, the prompt now shows:
- The fastest reachable transport mode and estimated ETA
- A `✓ REACHABLE` or `✗ NOT REACHABLE` flag with the reason

Example output for R102 vs R103:
```
→ R103 | urgency=1B | ✓ REACHABLE via charter ~68min
→ R102 | urgency=1B | ✗ NOT REACHABLE — fastest (charter) ~195min > viability 180min
```

The agent now correctly chose R103 (same hospital, reachable) on the next run.

---

## 7. Results and Evaluation

### 7.1 Final Scores

Evaluated using Groq `llama-3.3-70b-versatile`, seed=42:

| Task | Difficulty | Our Score | GPT-4o-mini Baseline | Improvement |
|------|------------|-----------|----------------------|-------------|
| Easy | easy | **1.000** | 0.780 | +0.220 |
| Medium | medium | **0.908** | 0.520 | +0.388 |
| Hard | hard | **0.842** | 0.310 | +0.532 |
| **Mean** | | **0.917** | **0.540** | **+0.377** |

### 7.2 Component Analysis

**Easy task (1.000)**  
Perfect score across all five components. The agent correctly identified the Status 1A recipient with compatible blood type, matched in step 1, dispatched in step 2, and the episode terminated in 2 steps — the minimum possible.

**Medium task (0.908)**  
Four of five components scored 1.0. The single miss was `urgency_priority = 0.333`, meaning only 1 of 3 Status 1A recipients was matched. The agent correctly matched the heart (D002) to R010 (1A) and dispatched it. It then correctly requested a crossmatch on R020 (1A liver, MELD=35) and received a negative result (safe to proceed). However, it then matched R021 (1B, MELD=28) instead of R020. This is a reasoning gap: the crossmatch result appeared in a separate step, and the model did not carry the clearance forward into its matching decision.

**Hard task (0.842)**  
`urgency_priority = 0.0` is correct and unavoidable — both Status 1A recipients (R101 for lung, R110 for heart) had positive crossmatches indicating hyperacute rejection risk. The agent correctly identified and skipped both. `quality = 0.770` is lower than the other tasks because the safe recipients had lower compatibility scores than the 1A patients would have had — also unavoidable by task design. All three organs were matched, dispatched, and no organ was wasted.

### 7.3 Comparison to Expected

The original codebase documented expected scores of ~0.72 / ~0.48 / ~0.25 for easy, medium, hard using Qwen2.5-1.5B on the HF API. Our improvements to the observation prompt, system prompt, and model selection produced scores of 1.000 / 0.908 / 0.842 — substantially above both the original expected values and the GPT-4o-mini baseline.

---

## 8. Known Limitations

### 8.1 Medium Task Reasoning Gap
The agent passes the crossmatch on R020 but then selects R021 for the liver match. This is a consistent failure mode related to multi-step reasoning continuity. The crossmatch result and the subsequent match decision are separated by at least one step, and the model does not reliably treat a negative crossmatch result as a match recommendation for that specific pair.

### 8.2 Fixed Task Scenarios
The three tasks are hardcoded scenarios. The agent could in principle memorise optimal action sequences rather than genuinely learning allocation reasoning. Generating randomised task instances from the NHS statistics would provide a more rigorous evaluation.

### 8.3 No Real Crossmatch Result Propagation
The environment records that a crossmatch was requested but does not automatically block the agent from attempting to match a pair that returned a positive crossmatch. The model is expected to reason from the textual result in its history. A more robust design would have the environment enforce positive crossmatch results as hard blocks.

### 8.4 Single-Process Server State
The FastAPI server holds all episode state in an in-memory dictionary. Restarting the server loses all active episodes. This is acceptable for local development but unsuitable for production deployment or distributed training.

### 8.5 Heuristic GRPO Reward Function
The GRPO training reward function uses a stateless heuristic (correct JSON format, correct fields per action type) rather than running the full environment. This means the training signal does not capture clinical correctness — it only rewards well-formed actions. A fully environment-coupled training loop would produce better-calibrated rewards.

---

## 9. Future Work

### 9.1 Fix the Medium Task Reasoning Gap
Explicitly mark crossmatch-cleared recipients in the observation prompt (e.g. `[crossmatch NEGATIVE — CLEARED FOR MATCHING]`). This would remove ambiguity between "crossmatch requested" and "crossmatch cleared" and close the gap between step 3 and step 4 reasoning.

### 9.2 Randomised Task Generation
Extend `NHSDataLoader` to generate synthetic tasks by sampling donors and recipients from the NHS statistics distributions. This would allow training on a much larger and more varied set of scenarios, reducing the risk of memorisation and improving generalisation.

### 9.3 Environment-Coupled GRPO Training
Replace the heuristic reward function in `train_grpo.py` with a full environment rollout inside the reward function. This would give the GRPO trainer clinically accurate reward signals at the cost of higher compute per training step.

### 9.4 Enforce Positive Crossmatch as Hard Block
Update `_do_match` in `environment.py` to check whether a positive crossmatch has already been recorded for the donor-recipient pair and return a penalty if matching is attempted. This would close the safety gap where the model could, in principle, proceed after a positive result.

### 9.5 Extended Hospital Network
The current 15-hospital network covers all designated UK NHSBT transplant centres at city level. Adding ward-level coordinates and incorporating actual NHS transport contracts (e.g. HEMS helicopter coverage zones) would improve transport time accuracy.

### 9.6 Additional Clinical Protocols
- **Paediatric allocation** — under-18 recipients have different priority rules
- **DCD-specific viability reductions** — DCD organs have shorter effective viability than DBD; the current model approximates this only for the hard task lung
- **UKELD scoring** — the current implementation maps UKELD ≈ MELD for simplicity; a proper UKELD calculation would improve liver allocation accuracy

---

## 10. Conclusion

This project demonstrates that an LLM agent, given well-structured observations and a carefully designed prompt, can achieve near-optimal performance on a complex, multi-step, clinically constrained decision-making task. The key insights from the development process are:

1. **Observation design is the most impactful lever.** The difference between a score of −1.68 (looping agent, no transport feasibility) and +3.05 (near-perfect hard task) came almost entirely from improving what the model was shown, not from changing the model itself.

2. **Explicit workflow state beats implicit reasoning.** Telling the model "NEXT REQUIRED ACTION: dispatch_transport" was more effective than hoping it would infer the correct step from context alone.

3. **Pre-filtering compatible options reduces error surface.** Grouping recipients under their compatible donor, and marking unreachable options with transport feasibility flags, eliminated entire categories of mistakes (wrong organ type, impossible transit times) without requiring the model to do that reasoning itself.

4. **Free inference backends are now viable for research.** Groq's free tier delivered ~1s/step inference on a 70B parameter model, making it a practical alternative to paid API access for evaluation and prototyping.

5. **Protocol violations need to be visible, not just penalised.** Making NHSBT rules explicit in the system prompt — and making the consequences of violating them visible in the observation — produced safer and more protocol-compliant agent behaviour than relying on negative rewards alone.

The mean aggregate score of **0.917** against a **0.540** baseline, using a free inference backend, demonstrates that LLM agents are a credible approach to complex clinical logistics problems — and that prompt engineering and observation design are at least as important as model scale for this class of task.
