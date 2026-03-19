"""Pydantic models for the AI work token pipeline.

These models define the common schema for representing units of AI work,
submissions, evaluations, and reward decisions across all services.
"""

from time import time
from typing import Dict, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class AIWorkTask(BaseModel):
    """A unit of AI work to be performed by an agent."""

    task_id: str
    description: str
    requester_address: str
    reward_amount: float = Field(gt=0)
    created_at: float = Field(default_factory=time)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def task_id_must_be_nonempty(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return v


class AIWorkSubmission(BaseModel):
    """A submission of completed AI work by an agent."""

    submission_id: str
    task_id: str
    worker_address: str
    result: str
    submitted_at: float = Field(default_factory=time)
    trace_metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator("submission_id", "task_id")
    @classmethod
    def ids_must_be_nonempty(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return v


class AIWorkEvaluation(BaseModel):
    """Evaluation outcome for a submitted piece of AI work."""

    evaluation_id: str
    submission_id: str
    evaluator_address: str
    raw_score: float
    normalized_score: float = Field(ge=0.0, le=1.0)
    comments: str = ""
    evaluated_at: float = Field(default_factory=time)

    @field_validator("evaluation_id", "submission_id")
    @classmethod
    def ids_must_be_nonempty(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return v


class RewardDecision(BaseModel):
    """Decision on whether and how much to reward for evaluated AI work."""

    decision_id: str
    evaluation_id: str
    task_id: str
    worker_address: str
    reward_amount: float = Field(ge=0)
    approved: bool
    reason: Optional[str] = None
    decided_at: float = Field(default_factory=time)

    @field_validator("decision_id", "evaluation_id", "task_id")
    @classmethod
    def ids_must_be_nonempty(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return v
