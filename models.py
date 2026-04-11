from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    """Represents an agent action passed into environment.step()."""

    action: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Represents current environment state returned to the agent.

    openenv-core requires that step() / reset() return an Observation that
    embeds ``reward`` (float) and ``done`` (bool) directly.  These two fields
    are stripped from the inner observation payload by serialize_observation()
    and surfaced as top-level fields in the HTTP response.
    """

    task_id: str = ""
    task_level: str = ""
    app_id: str = ""
    step_count: int = 0
    max_steps: int = 0
    required_actions: List[str] = Field(default_factory=list)
    completed_actions: List[str] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    results: Dict[str, Any] = Field(default_factory=dict)
    # Data-fetch quality metrics set at reset() time.
    data_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    fetch_reward: float = Field(default=0.0)
    # Required by openenv-core's serialize_observation():
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    error: Optional[str] = Field(default=None)


class Reward(BaseModel):
    """Represents incremental and cumulative reward."""

    delta: float
    total: float
    reason: str
    penalties: List[str] = Field(default_factory=list)
