"""
Transplant Logistics Environment — FastAPI Server
Exposes the OpenEnv HTTP interface:
  POST /reset
  POST /step
  GET  /state
  GET  /health
  GET  /tasks
  POST /grade
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

from models import (
    StepResult, TransplantAction, TransplantObservation, TransplantState,
)
from server.environment import TransplantEnv, TransplantGrader, TASKS

app = FastAPI(
    title="Transplant Logistics OpenEnv — NHS UK",
    description=(
        "Real-world organ allocation and transplant logistics environment "
        "for RL agent training. Implements the OpenEnv step()/reset()/state() API. "
        "Calibrated to NHS Blood and Transplant (NHSBT) 2022/23 data."
    ),
    # FIX: version was "1.0.0" — run_guide expects "2.0.0"
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per task (stateful)
_envs: Dict[str, TransplantEnv] = {}
_grader = TransplantGrader()


# ── Request schemas ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy_clear_match"
    seed: int    = 42

class StepRequest(BaseModel):
    task_id: str
    action:  TransplantAction

class GradeRequest(BaseModel):
    task_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "environment": "transplant-logistics-env-nhs",
        "version":     "2.0.0",
    }


@app.get("/tasks")
def list_tasks():
    return {
        tid: {
            "name":        t["name"],
            "difficulty":  t["difficulty"],
            "description": t["description"],
            "max_steps":   t["max_steps"],
        }
        for tid, t in TASKS.items()
    }


@app.post("/reset", response_model=TransplantObservation)
def reset(req: ResetRequest):
    if req.task_id not in TASKS:
        raise HTTPException(404, f"Unknown task_id: {req.task_id}")
    env = TransplantEnv(req.task_id)
    _envs[req.task_id] = env
    return env.reset(seed=req.seed)


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    env = _envs.get(req.task_id)
    if env is None:
        raise HTTPException(400, "Call /reset first")
    return env.step(req.action)


@app.get("/state", response_model=TransplantState)
def state(task_id: str = "task_easy_clear_match"):
    env = _envs.get(task_id)
    if env is None:
        raise HTTPException(400, "Call /reset first")
    return env.state()


@app.post("/grade")
def grade(req: GradeRequest) -> Dict[str, Any]:
    env = _envs.get(req.task_id)
    if env is None:
        raise HTTPException(400, "Call /reset first")
    s      = env.state()
    task   = TASKS[req.task_id]
    scores = _grader.grade(s, task)
    return {"task_id": req.task_id, "scores": scores}


@app.get("/")
def root():
    return {
        "name":      "Transplant Logistics OpenEnv — NHS UK",
        "version":   "2.0.0",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/grade"],
        "docs":      "/docs",
    }
