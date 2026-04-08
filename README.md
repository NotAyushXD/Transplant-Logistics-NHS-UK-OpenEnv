---
title: Transplant Logistics OpenEnv
emoji: 🫀
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
tags:
  - openenv
  - healthcare
  - reinforcement-learning
---

# 🫀 Transplant Logistics — NHS UK OpenEnv

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/openenv-v1-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)

A **Reinforcement Learning environment** for training and evaluating AI agents on real-world organ transplant allocation decisions, calibrated to NHS Blood and Transplant (NHSBT) 2022/23 data.

The agent acts as a national transplant coordinator — matching donor organs to recipients based on blood compatibility, HLA typing, organ viability windows, transport logistics, and clinical urgency, under severe time pressure.

> 🏥 **~6,959 patients** were on the UK organ transplant waiting list at end March 2023.  
> **~439 patients** die each year while waiting. Every allocation decision matters.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Usage](#usage)
- [Tasks](#tasks)
- [Results](#results)
- [Configuration](#configuration)
- [Training and Evaluation Pipeline](#training-and-evaluation-pipeline)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---
<<<<<<< HEAD
# 🫀 Transplant Logistics — NHS UK OpenEnv

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/openenv-v1-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)

A **Reinforcement Learning environment** for training and evaluating AI agents on real-world organ transplant allocation decisions, calibrated to NHS Blood and Transplant (NHSBT) 2022/23 data.

The agent acts as a national transplant coordinator — matching donor organs to recipients based on blood compatibility, HLA typing, organ viability windows, transport logistics, and clinical urgency, under severe time pressure.

> 🏥 **~6,959 patients** were on the UK organ transplant waiting list at end March 2023.  
> **~439 patients** die each year while waiting. Every allocation decision matters.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Usage](#usage)
- [Tasks](#tasks)
- [Results](#results)
- [Configuration](#configuration)
- [Training and Evaluation Pipeline](#training-and-evaluation-pipeline)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---
=======
>>>>>>> 1e66112b55a7c0dd5d010cd4f57c33685fac1266

## Features

- 🏥 **15 real UK NHSBT transplant centres** with accurate coordinates and transport modelling
- 🩸 **Full NHSBT protocol enforcement** — ABO compatibility, HLA crossmatching, PRA thresholds, cold ischaemia limits
- 📊 **Calibrated to NHS 2022/23 data** via the [Kaggle NHS Organ Donation dataset](https://www.kaggle.com/datasets/patricklford/nhs-organ-donation)
- ⚡ **Dense reward function** — feedback on every action, not just episode end
- 🤖 **Three inference backends** — Groq (free), HuggingFace API, or local model weights
- 🎓 **GRPO fine-tuning** — post-training script using TRL's GRPOTrainer
- 🌐 **OpenEnv-compliant HTTP API** — FastAPI server with Swagger UI

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          RL LOOP                             │
│                                                              │
│  [ LLM Agent ]  ←→  [ FastAPI Server ]  ←→  [ Simulation ]  │
│  inference_hf.py      app.py               environment.py   │
│                                                              │
│  Reads observation    Routes HTTP calls    Holds state,     │
│  Decides action       to env instances     enforces rules,  │
│  Parses/sends JSON                         computes reward  │
└──────────────────────────────────────────────────────────────┘
```

---

## Quickstart

```bash
# 1. Clone and set up
<<<<<<< HEAD
git clone https://github.com/your-username/transplant-env.git
cd transplant-env
=======
git clone https://huggingface.co/spaces/Thanya710/transplant-logistics-env
cd transplant-logistics-env
>>>>>>> 1e66112b55a7c0dd5d010cd4f57c33685fac1266
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Create package markers
touch server/__init__.py baseline/__init__.py training/__init__.py

# 3. Get a free Groq API key at https://console.groq.com
export GROQ_API_KEY=gsk_your_key_here

# 4. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload &

# 5. Run the agent
python baseline/inference_hf.py --backend groq
```

---

## Installation

### Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or 3.11 |
| RAM | ≥ 8 GB (16 GB for local inference) |
| GPU | Optional (required for GRPO training) |

### Steps

**1. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

**2. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Create Python package markers**

```bash
touch server/__init__.py
touch baseline/__init__.py
touch training/__init__.py
```

**4. (Optional) Download the NHS Kaggle dataset**

The CSV calibrates wait times and utilisation statistics with real NHS data. The environment works without it using built-in 2022/23 fallback statistics.

```bash
# Add your Kaggle credentials first: kaggle.com → Settings → API → Create Token
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d patricklford/nhs-organ-donation
unzip nhs-organ-donation.zip -d data/nhs/
```

**5. Smoke test**

```bash
python nhs_data_explorer.py --csv data/nhs/NHS_Organ_Donation.csv
# → Pipeline status: ✓ ALL TASKS PASSED
```

---

## Usage

### Start the API server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Verify:
```bash
curl http://localhost:7860/health
# → {"status":"ok","environment":"transplant-logistics-env-nhs","version":"2.0.0"}
```

Interactive API docs available at http://localhost:7860/docs.

### Run the agent

**Groq (recommended — free, ~1s/step)**

```bash
# Get a free key at https://console.groq.com
export GROQ_API_KEY=gsk_your_key_here

python baseline/inference_hf.py --backend groq --csv data/nhs/NHS_Organ_Donation.csv
```

**HuggingFace local model (no account needed)**

```bash
python baseline/inference_hf.py --backend local --model Qwen/Qwen2.5-1.5B-Instruct
```

**HuggingFace Inference API**

> ⚠️ Most popular models now route through paid providers on the HF Inference API. Use Groq instead.

```bash
export HF_TOKEN=hf_your_token_here
python baseline/inference_hf.py --backend api --model Qwen/Qwen2.5-1.5B-Instruct
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `groq` | `groq` / `api` / `local` |
| `--model` | *(see below)* | Model ID. Groq default: `llama-3.3-70b-versatile` |
| `--task` | `all` | `task_easy_clear_match` / `task_medium_cascade_allocation` / `task_hard_expiry_crisis` / `all` |
| `--seed` | `42` | Random seed for environment reset |
| `--csv` | `None` | Path to NHS Kaggle CSV file |
| `--quiet` | off | Only print final score table |
| `--history-turns` | `4` | Max conversation turns kept in context. Reduce to `2` if hitting 413 errors |

---

## Tasks

### 🟢 Easy — Single Kidney, Clear Best Match

One donor kidney, three recipients. The correct match is unambiguous: compatible blood type, Status 1A urgency, same hospital, low PRA, long wait time. Tests baseline allocation reasoning.

**Required actions:** `match_organ` → `dispatch_transport` → `notify_team`

### 🟡 Medium — Multi-Organ Cascade (Heart + Liver)

Two organs from the same DBD donor. The heart has **4h viability**, the liver **12h**. The heart must be matched and dispatched first or it expires. One heart recipient has PRA=88% — a trap for agents that naively prioritise urgency without checking sensitisation.

**Required actions:** match heart → dispatch → match liver → dispatch → `notify_team`

### 🔴 Hard — Expiry Crisis with Misleading Signals

Three donors, five recipients, 20 steps. A DCD lung expires within 3 steps. The Status 1A heart recipient has a **positive virtual crossmatch** (hyperacute rejection risk — do not match). A high-KDPI=0.85 kidney should go to the older recipient. Requires crossmatch protocol, PRA reasoning, and KDPI age-matching simultaneously.

**Required actions:** crossmatch → match lung → dispatch → crossmatch → match heart → dispatch → match kidney → dispatch → `notify_team`

---

## Results

Observed scores using **Groq `llama-3.3-70b-versatile`**, seed=42:

| Task | Difficulty | Score | vs GPT-4o-mini baseline |
|------|------------|-------|------------------------|
| Single kidney | easy | **1.000** | +0.22 |
| Multi-organ cascade | medium | **0.908** | +0.39 |
| Expiry crisis | hard | **0.842** | +0.53 |
| **Mean** | | **0.917** | **+0.38** |

### Grade components (0.0–1.0)

| Component | Weight | Meaning |
|-----------|--------|---------|
| `transplant_rate` | 35% | Fraction of available organs successfully matched |
| `quality` | 25% | Mean compatibility score across accepted matches |
| `viability_margin` | 20% | Fraction of transplants with ≥30 min viability buffer |
| `urgency_priority` | 10% | Fraction of Status 1A patients matched |
| `safety` | 10% | No dangerous high-PRA matches without prior crossmatch |

> **Note:** `urgency_priority = 0.0` on the hard task is **correct behaviour** — both Status 1A recipients have positive crossmatches (hyperacute rejection risk) and legally cannot be matched. The agent handles this correctly.

---

## Configuration

### Groq model selection

| Model | TPM limit | Speed | Recommended for |
|-------|-----------|-------|-----------------|
| `llama-3.3-70b-versatile` | 12,000 | ~1s/step | **Default — all tasks** |
| `llama-3.1-8b-instant` | 6,000 | ~0.5s/step | Easy/medium only (hits 413 on hard task) |
| `mixtral-8x7b-32768` | 5,000 | ~1s/step | Long context experiments |

> ⚠️ Do not use `llama-3.1-8b-instant` for the hard task. Its 6,000 TPM limit is exceeded by the growing conversation history. The code auto-retries with shorter history, but `llama-3.3-70b-versatile` avoids this entirely.

### Context window (`--history-turns`)

The agent carries its full conversation history across steps so it can reason about past actions. By default, only the last 4 exchanges are kept to stay within model context limits. The full environment state is always available in the current observation, so dropping older turns does not lose critical information.

```bash
# Reduce if hitting 413 on smaller models
python baseline/inference_hf.py --backend groq --history-turns 2
```

### NHSBT Protocol constants (`server/environment.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `NHSBT_CROSSMATCH_PRA_THRESHOLD` | 0.85 | PRA above this requires mandatory crossmatch before matching |
| `MINUTES_PER_STEP` | 30 | Simulated time per environment step |

Cold ischaemia limits:

| Organ | Limit |
|-------|-------|
| Heart | 4 hours |
| Lung | 6 hours |
| Liver | 12 hours |
| Kidney | 24 hours |
| Pancreas | 12 hours |

---

## Training and Evaluation Pipeline

A standard workflow is to first evaluate a strong API baseline (like Groq) to measure zero-shot agent performance, then train a smaller local model using **GRPO (Group Relative Policy Optimisation)**, and finally evaluate the fine-tuned local model.

### 1. Establish Baselines via Groq API

You can run the `train_grpo.py` script directly with Groq models to quickly collect rollout trajectories and score their performance. 
*(Note: Supplying an API model seamlessly skips the PyTorch GRPO weight update "Phase 2" since we cannot alter remote model weights).*

```bash
# Requires GROQ_API_KEY environment variable. 
# Automatically jumps to Phase 1 (rollouts) and Phase 3 (evaluation).
python training/train_grpo.py \
    --model llama-3.3-70b-versatile \
    --backend groq \
    --output ./checkpoints/transplant-grpo
```

### 2. Fine-tune a Local Model via GRPO

Once you have baselines, you can train a smaller, open weights model locally using the environment directly. This requires an environment capable of computing gradients (PyTorch).

> **GPU required.** CPU training for the underlying policy model is too slow to be practical.

```bash
# API server must be running (see Usage above)
python training/train_grpo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --backend hf \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 4 \
    --lr 5e-6 \
    --rollouts-per-task 4 \
    --num-generations 4 \
    --output ./checkpoints/transplant-grpo
```

### 3. Evaluate the Fine-Tuned Model

Once your model checkpoints are saved, you can run the inference script over them to thoroughly observe exactly what decisions the updated weights propose.

```bash
python baseline/inference_hf.py \
    --backend local \
    --model ./checkpoints/transplant-grpo \
    --csv data/nhs/NHS_Organ_Donation.csv
```

---

## How It Works

### Simulation engine (`server/environment.py`)

Every call to `env.step(action)` runs four things in sequence:

```
1. Advance time          +30 min cold ischaemia on every donor
2. Execute the action    compute reward based on NHSBT rules
3. Check organ expiry    remove expired organs, apply waste penalty
4. Check done condition  max steps / all donors gone / notify_team called
```

**Compatibility scoring** starts at 0.5 and adjusts based on:
- HLA crossreactivity (−0.08 per shared antigen/antibody pair)
- PRA sensitisation (−0.15 × PRA)
- Urgency tier (+0.30 for 1A, +0.18 for 1B, +0.05 for routine)
- Waiting time (+up to 0.15, log-scaled)
- KDPI age-match bonus for kidneys (±0.10)
- MELD/UKELD severity for liver (+up to 0.10)

Returns **0.0 immediately** for blood-type or organ-type mismatch.

**Transport times** use the Haversine formula between real UK hospital coordinates plus mode-specific speed and overhead (ground ambulance 70 km/h, air charter 280 km/h, commercial 500 km/h).

### LLM agent (`baseline/inference_hf.py`)

```
obs = env.reset()
history = []

while not done:
    1. obs_to_prompt(obs)         → structured text with transport feasibility
    2. [system] + history + text  → LLM backend
    3. raw text                   → parse_action() → TransplantAction JSON
    4. env.step(action)           → reward + next observation
    5. append (user, assistant)   → history (last N turns kept)

grader.grade() → final 0–1 score
```

The observation prompt explicitly shows:
- A **workflow state banner** telling the model what action comes next
- **Compatible recipients grouped under each donor** (pre-filtered by blood type and organ type)
- **Transport feasibility per recipient** — estimated ETA for each transport mode with a ✓/✗ reachability flag
- A **DO NOT ATTEMPT** section listing incompatible recipients

---

## File Structure

```
transplant-env/
├── PROCESS.md                 Development process logging
├── Dockerfile                 Container deployment configuration
├── models.py                  Pydantic data contracts (all shared types)
├── client.py                  HTTP client wrapper for training loops
├── nhs_data_explorer.py       NHS data inspection + heuristic smoke test
├── smoke_test.py              Minimal import + reset test
├── openenv.yaml               OpenEnv framework metadata
├── requirements.txt
├── data/
│   └── nhs/                   Kaggle CSV goes here
├── server/
│   ├── __init__.py
│   ├── app.py                 FastAPI server (HTTP interface)
│   └── environment.py         Core simulation, compatibility, reward, grader
├── baseline/
│   ├── __init__.py
│   └── inference_hf.py        LLM agent (Groq / HF API / local backends)
└── training/
    ├── __init__.py
    └── train_grpo.py          GRPO fine-tuning loop (TRL)
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `413 Request Too Large` | Conversation history exceeded model TPM limit | Switch to `llama-3.3-70b-versatile` or use `--history-turns 2` |
| `402 Payment Required` (HF) | HF free credits exhausted | Switch to `--backend groq` (free) |
| `ModuleNotFoundError: No module named 'server'` | Missing `__init__.py` or wrong working directory | `touch server/__init__.py baseline/__init__.py training/__init__.py` and run from project root |
| `ValidationError` for `Hospital` / `nhs_trust` | Original `models.py` without fix | Add `nhs_trust: Optional[str] = None` to `Hospital` class in `models.py` |
| `503 Service Unavailable` (HF) | Serverless endpoint cold-starting | Wait 30s and retry |
| `parse error` — no JSON found | Model returned non-JSON output | Use a larger model: `--model llama-3.3-70b-versatile` |
| `OutOfMemoryError` (local) | Model too large for available VRAM | Use `--model Qwen/Qwen2.5-1.5B-Instruct` or `--backend groq` |
| `{"detail": "Call /reset first"}` | Server restarted, episode state lost | `curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task_easy_clear_match","seed":42}'` |

---

## Contributing

Contributions are welcome. Some areas where improvements would have the most impact:

- **Medium task reasoning gap** — the agent correctly crossmatches R020 (1A liver, MELD=35) and gets a negative result, but then matches R021 (1B) instead. The prompt could be updated to mark crossmatch-cleared recipients more explicitly.
- **Synthetic task generation** — extending `NHSDataLoader` to generate randomised scenarios from the NHS statistics for training diversity.
- **Additional NHSBT protocols** — UKELD scoring, DCD-specific viability reductions, paediatric allocation rules.
- **New tasks** — pancreas allocation, multi-centre cascade, deceased-donor kidney sharing.

Please open an issue before submitting a large pull request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

NHS statistics used for calibration are sourced from the publicly available
[NHS Blood and Transplant Activity Report 2022/23](https://www.nhsbt.nhs.uk/what-we-do/transplantation-services/statistics-and-clinical-studies/activity-reports/).
<<<<<<< HEAD

=======
>>>>>>> 1e66112b55a7c0dd5d010cd4f57c33685fac1266
