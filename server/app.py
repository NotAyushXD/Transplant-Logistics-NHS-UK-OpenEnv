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

from fastapi import FastAPI, HTTPException, Request
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

class StepRequest(BaseModel):
    task_id: str
    action:  TransplantAction


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


@app.post("/reset")
async def reset(request: Request):
    """Reset the environment. Accepts empty body (uses defaults) or JSON with task_id/seed."""
    try:
        body = await request.json()
        if body is None:
            body = {}
    except Exception:
        body = {}

    task_id = body.get("task_id", "task_easy_clear_match")
    seed = body.get("seed", 42)

    if task_id not in TASKS:
        raise HTTPException(404, f"Unknown task_id: {task_id}")
    env = TransplantEnv(task_id)
    _envs[task_id] = env
    obs = env.reset(seed=seed)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    env = _envs.get(req.task_id)
    if env is None:
        raise HTTPException(400, "Call /reset first")
    result = env.step(req.action)
    return result.model_dump()


@app.get("/state")
def state(task_id: str = "task_easy_clear_match"):
    env = _envs.get(task_id)
    if env is None:
        raise HTTPException(400, "Call /reset first")
    s = env.state()
    return s.model_dump()


@app.post("/grade")
async def grade(request: Request) -> Dict[str, Any]:
    """Grade the current episode. Accepts JSON with task_id or empty body (uses default)."""
    try:
        body = await request.json()
        if body is None:
            body = {}
    except Exception:
        body = {}

    task_id = body.get("task_id", "task_easy_clear_match")

    env = _envs.get(task_id)
    if env is None:
        raise HTTPException(400, "Call /reset first")
    s      = env.state()
    task   = TASKS[task_id]
    scores = _grader.grade(s, task)
    return {"task_id": task_id, "scores": scores}


@app.get("/")
def root():
    return {
        "name":      "Transplant Logistics OpenEnv — NHS UK",
        "version":   "2.0.0",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/grade"],
        "docs":      "/docs",
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
