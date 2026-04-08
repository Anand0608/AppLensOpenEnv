from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Action(BaseModel):
    """Represents an agent action passed into environment.step()."""

    action: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Represents current environment state returned to the agent."""

    task_id: str
    task_level: str
    app_id: str
    step_count: int
    max_steps: int
    required_actions: List[str]
    completed_actions: List[str]
    available_actions: List[str]
    results: Dict[str, Any] = Field(default_factory=dict)
    # Data-fetch quality metrics set at reset() time.
    data_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    fetch_reward: float = Field(default=0.0)


class Reward(BaseModel):
    """Represents incremental and cumulative reward."""

    delta: float
    total: float
    reason: str
    penalties: List[str] = Field(default_factory=list)
