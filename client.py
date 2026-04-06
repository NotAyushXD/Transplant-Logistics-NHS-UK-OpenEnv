"""
Transplant Logistics — Python Client
Wraps the HTTP API for use in RL training loops.

Usage:
    from client import TransplantClient
    from models import TransplantAction, ActionType

    async with TransplantClient("http://localhost:7860") as client:
        obs  = await client.reset("task_easy_clear_match")
        result = await client.step("task_easy_clear_match",
            TransplantAction(action_type=ActionType.MATCH_ORGAN,
                             donor_id="D001", recipient_id="R001"))
        state  = await client.state("task_easy_clear_match")
        grades = await client.grade("task_easy_clear_match")
"""

from __future__ import annotations

import httpx
from typing import Any, Dict, Optional

from models import (
    StepResult, TransplantAction, TransplantObservation, TransplantState,
)


class TransplantClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *_):
        if self._client:
            await self._client.aclose()

    # ── sync convenience (for notebooks / scripts) ────────────────────────

    def reset_sync(self, task_id: str, seed: int = 42) -> TransplantObservation:
        resp = httpx.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=15,
        )
        resp.raise_for_status()
        return TransplantObservation.model_validate(resp.json())

    def step_sync(self, task_id: str,
                  action: TransplantAction) -> StepResult:
        resp = httpx.post(
            f"{self.base_url}/step",
            json={"task_id": task_id, "action": action.model_dump()},
            timeout=15,
        )
        resp.raise_for_status()
        return StepResult.model_validate(resp.json())

    def state_sync(self, task_id: str) -> TransplantState:
        resp = httpx.get(
            f"{self.base_url}/state",
            params={"task_id": task_id},
            timeout=15,
        )
        resp.raise_for_status()
        return TransplantState.model_validate(resp.json())

    def grade_sync(self, task_id: str) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/grade",
            json={"task_id": task_id},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def tasks_sync(self) -> Dict[str, Any]:
        resp = httpx.get(f"{self.base_url}/tasks", timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ── async ─────────────────────────────────────────────────────────────

    async def reset(self, task_id: str, seed: int = 42) -> TransplantObservation:
        resp = await self._client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        resp.raise_for_status()
        return TransplantObservation.model_validate(resp.json())

    async def step(self, task_id: str,
                   action: TransplantAction) -> StepResult:
        resp = await self._client.post(
            f"{self.base_url}/step",
            json={"task_id": task_id, "action": action.model_dump()},
        )
        resp.raise_for_status()
        return StepResult.model_validate(resp.json())

    async def state(self, task_id: str) -> TransplantState:
        resp = await self._client.get(
            f"{self.base_url}/state",
            params={"task_id": task_id},
        )
        resp.raise_for_status()
        return TransplantState.model_validate(resp.json())

    async def grade(self, task_id: str) -> Dict[str, Any]:
        resp = await self._client.post(
            f"{self.base_url}/grade",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        return resp.json()
