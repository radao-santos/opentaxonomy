"""
Intermediate Pydantic schemas for LLM responses.
These are internal to the LLM layer and not part of the public taxonomy output format.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel

from ..taxonomy.models import EdgeCase, DecisionRecord


# ── Normalization ──────────────────────────────────────────────────────────────

class NormalizedEntity(BaseModel):
    normalized: str
    raw_samples: list[str]


class NormalizationResult(BaseModel):
    entities: list[NormalizedEntity]


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
