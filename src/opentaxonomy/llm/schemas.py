"""
Intermediate Pydantic schemas for LLM responses.
These are internal to the LLM layer and not part of the public taxonomy output format.
"""
from __future__ import annotations

import json
from typing import Any, Optional
from pydantic import BaseModel, field_validator

from ..taxonomy.models import EdgeCase, DecisionRecord


def _parse_stringified_list(v: Any) -> Any:
    """LLMs occasionally return a JSON array as a string. Parse it back."""
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return v


# ── Normalization ──────────────────────────────────────────────────────────────

class NormalizedEntity(BaseModel):
    normalized: str
    definition: str
    raw_samples: list[str]


class NormalizationResult(BaseModel):
    entities: list[NormalizedEntity]

    _parse_entities = field_validator("entities", mode="before")(_parse_stringified_list)


# ── Foundation (Q0 + Q0b) ─────────────────────────────────────────────────────

class FoundationResult(BaseModel):
    root_label: str
    context_Q0: str
    context_Q0b: str
    seed_id: str
    canonical_prefix: str
    description: str
    normalization_rules: list[str]


# ── Primary differentiation (Q1) ──────────────────────────────────────────────

class BranchDef(BaseModel):
    key: str
    label: str
    dimension: str
    question: str
    members: list[str]
    criteria_includes: list[str]
    criteria_excludes: list[str]


class PrimaryDiffResult(BaseModel):
    dimension: str
    question: str
    branches: list[BranchDef]
    unplaced: list[str] = []

    _parse_branches = field_validator("branches", mode="before")(_parse_stringified_list)
    _parse_unplaced = field_validator("unplaced", mode="before")(_parse_stringified_list)


# ── Recursive differentiation (Q2) ────────────────────────────────────────────

class RecursiveDiffResult(BaseModel):
    is_leaf: bool
    leaf_question: Optional[str] = None
    leaf_criteria_includes: Optional[list[str]] = None
    leaf_criteria_excludes: Optional[list[str]] = None
    sub_branches: Optional[list[BranchDef]] = None
    unplaced: list[str] = []


# ── Node detail generation ────────────────────────────────────────────────────

class NodeDetailResult(BaseModel):
    question: str
    criteria_includes: list[str]
    criteria_excludes: list[str]
    edge_cases: list[EdgeCase] = []
    decision_record: Optional[DecisionRecord] = None


# ── Tree traversal (expand) ───────────────────────────────────────────────────

class ValueFit(BaseModel):
    value: str
    matching_canonical_id: Optional[str]  # None = doesn't fit any existing child
    reasoning: str


class TreeFitResult(BaseModel):
    fits: list[ValueFit]

    _parse_fits = field_validator("fits", mode="before")(_parse_stringified_list)


class GraftBranch(BaseModel):
    key: str
    label: str
    question: str
    criteria_includes: list[str]
    criteria_excludes: list[str]
    members: list[str]
    edge_cases: list[EdgeCase] = []


class GraftResult(BaseModel):
    branches: list[GraftBranch]
    unplaced: list[str] = []

    _parse_branches = field_validator("branches", mode="before")(_parse_stringified_list)
    _parse_unplaced = field_validator("unplaced", mode="before")(_parse_stringified_list)


# ── Placement ─────────────────────────────────────────────────────────────────

class PlacedValue(BaseModel):
    normalized: str
    canonical_id: str
    raw_samples: list[str]


class UnresolvedVal(BaseModel):
    raw: str
    reason: str


class PlacementResult(BaseModel):
    placed: list[PlacedValue]
    unresolved: list[UnresolvedVal] = []

    _parse_placed = field_validator("placed", mode="before")(_parse_stringified_list)
    _parse_unresolved = field_validator("unresolved", mode="before")(_parse_stringified_list)
