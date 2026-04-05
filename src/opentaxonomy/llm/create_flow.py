"""
CreateFlow — executes the full Prima Seed protocol:
  Step 0  →  Normalize
  Q0/Q0b  →  Foundation (root + context)
  Q1      →  Primary differentiation
  Q2      →  Recursive differentiation (per branch)
  Q3      →  Dialectical check (unplaced values)
  Detail  →  Generate full node definitions
  Finalize → Build Seed, Node[], PlacementMap
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import yaml
from rich.console import Console

from ..taxonomy.models import (
    Node,
    NodeCriteria,
    Placement,
    PlacementEntity,
    PlacementMap,
    Seed,
    SeedContext,
    SeedEdgeCase,
    SeedLevel,
    UnresolvedValue,
)
from ..utils.canonical_id import slugify
from .client import TaxonomyLLM
from .prompts import (
    SYSTEM_PRIMA_SEED,
    dialectical_check_prompt,
    foundation_prompt,
    node_details_prompt,
    normalize_prompt,
    primary_differentiation_prompt,
    recursive_differentiation_prompt,
)
from .schemas import (
    BranchDef,
    FoundationResult,
    NodeDetailResult,
    NormalizationResult,
    PrimaryDiffResult,
    RecursiveDiffResult,
)

console = Console()

_NORMALIZE_BATCH = 50
_PLACEMENT_BATCH = 50


# ── Internal tree node ────────────────────────────────────────────────────────

@dataclass
class _Node:
    key: str
    label: str
    canonical_id: str
    parent_label: Optional[str]
    members: list[str] = field(default_factory=list)
    children: list["_Node"] = field(default_factory=list)
    is_leaf: bool = False
    # Filled during detail generation:
    question: str = ""
    criteria_includes: list[str] = field(default_factory=list)
    criteria_excludes: list[str] = field(default_factory=list)
    edge_cases: list = field(default_factory=list)
    decision_record: Optional[object] = None


# ── Flow ──────────────────────────────────────────────────────────────────────

class CreateFlow:
    def __init__(self, llm: TaxonomyLLM, min_branch_size: int = 2, max_depth: int = 4):
        self.llm = llm
        self.min_branch_size = min_branch_size
        self.max_depth = max_depth

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self, raw_values: list[str], domain_hint: str = ""
    ) -> tuple[Seed, list[Node], PlacementMap]:
        # Step 0: Normalize
        console.print("[bold blue]Step 0:[/] Normalizing raw values…")
        entities = self._normalize(raw_values, domain_hint)
        normalized_unique = [e.normalized for e in entities]
        raw_by_norm: dict[str, list[str]] = {e.normalized: e.raw_samples for e in entities}
        def_by_norm: dict[str, str] = {e.normalized: e.definition for e in entities}
        console.print(f"  {len(raw_values)} raw -> {len(normalized_unique)} normalized entities")

        # Q0 + Q0b: Foundation
        console.print("[bold blue]Q0/Q0b:[/] Identifying form and context…")
        foundation = self._foundation(normalized_unique)
        prefix = foundation.canonical_prefix
        console.print(f"  Root: [italic]{foundation.root_label}[/]  prefix: [bold]{prefix}[/]")

        # Build root node
        root = _Node(
            key=prefix,
            label=foundation.root_label,
            canonical_id=prefix,
            parent_label=None,
            members=[],
        )

        # Q1: Primary differentiation
        console.print("[bold blue]Q1:[/] Primary differentiation…")
        l1 = self._primary_diff(normalized_unique, foundation.root_label, foundation.context_Q0b)
        for b in l1.branches:
            child = _build_child(b, parent=root)
            root.children.append(child)
        console.print(f"  {len(root.children)} L1 branches")

        # Q2: Recursive differentiation
        console.print("[bold blue]Q2:[/] Recursive differentiation…")
        all_nodes: list[_Node] = [root]

        def expand(node: _Node, depth: int) -> None:
            all_nodes.append(node)
            if depth >= self.max_depth or len(node.members) < self.min_branch_size:
                node.is_leaf = True
                return

            result = self._recursive_diff(node.label, node.members, foundation.context_Q0b, depth)

            if result.is_leaf:
                node.is_leaf = True
                if result.leaf_question:
                    node.question = result.leaf_question
                if result.leaf_criteria_includes:
                    node.criteria_includes = result.leaf_criteria_includes
                if result.leaf_criteria_excludes:
                    node.criteria_excludes = result.leaf_criteria_excludes
                return

            for b in (result.sub_branches or []):
                sub = _build_child(b, parent=node)
                node.children.append(sub)

            if not node.children:
                node.is_leaf = True
                return

            # Values not assigned to any sub-branch stay on this node as members
            assigned = {m for c in node.children for m in c.members}
            node.members = [m for m in node.members if m not in assigned]

            for child in node.children:
                expand(child, depth + 1)

        for l1_child in root.children:
            expand(l1_child, depth=2)

        # Q3: Dialectical check — collect unplaced values across all leaf nodes
        leaf_nodes = [n for n in all_nodes if n.is_leaf or not n.children]
        unplaced = _collect_unplaced(normalized_unique, leaf_nodes)
        q3_unresolved: list[UnresolvedValue] = []
        if unplaced:
            console.print(f"[bold blue]Q3:[/] Dialectical check — {len(unplaced)} unplaced values…")
            q3_unresolved = self._dialectical_check(unplaced, all_nodes, leaf_nodes, foundation.context_Q0b)

        # Generate full node definitions
        console.print("[bold blue]Detail:[/] Generating node definitions…")
        for node in all_nodes:
            if not node.question:
                detail = self._node_detail(node, foundation.context_Q0b)
                node.question = detail.question
                if not node.criteria_includes:
                    node.criteria_includes = detail.criteria_includes
                if not node.criteria_excludes:
                    node.criteria_excludes = detail.criteria_excludes
                node.edge_cases = detail.edge_cases
                node.decision_record = detail.decision_record

        # Assemble outputs
        seed = _build_seed(foundation, root, all_nodes)
        nodes = _build_output_nodes(all_nodes)
        placement_map = _build_placement_map(all_nodes, raw_by_norm, def_by_norm, foundation.seed_id, prefix)
        placement_map.unresolved.extend(q3_unresolved)

        return seed, nodes, placement_map

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _normalize(self, raw_values: list[str], domain_hint: str) -> list:
        all_entities = []
        for i in range(0, len(raw_values), _NORMALIZE_BATCH):
            batch = raw_values[i : i + _NORMALIZE_BATCH]
            result = self.llm.complete(
                NormalizationResult,
                system=SYSTEM_PRIMA_SEED,
                user=normalize_prompt(batch, domain_hint),
                max_tokens=8096,
            )
            all_entities.extend(result.entities)
        # Merge duplicate normalized names
        merged: dict[str, dict] = defaultdict(lambda: {"raws": [], "definition": ""})
        for e in all_entities:
            merged[e.normalized]["raws"].extend(e.raw_samples)
            if not merged[e.normalized]["definition"]:
                merged[e.normalized]["definition"] = e.definition
        from .schemas import NormalizedEntity
        return [NormalizedEntity(normalized=norm, definition=data["definition"], raw_samples=list(dict.fromkeys(data["raws"])))
                for norm, data in merged.items()]

    def _foundation(self, normalized: list[str]) -> FoundationResult:
        return self.llm.complete(
            FoundationResult,
            system=SYSTEM_PRIMA_SEED,
            user=foundation_prompt(normalized),
            max_tokens=2048,
        )

    def _primary_diff(self, normalized: list[str], root_label: str, context: str) -> PrimaryDiffResult:
        return self.llm.complete(
            PrimaryDiffResult,
            system=SYSTEM_PRIMA_SEED,
            user=primary_differentiation_prompt(normalized, root_label, context),
            max_tokens=8096,
        )

    def _recursive_diff(self, label: str, members: list[str], context: str, depth: int) -> RecursiveDiffResult:
        return self.llm.complete(
            RecursiveDiffResult,
            system=SYSTEM_PRIMA_SEED,
            user=recursive_differentiation_prompt(label, members, context, depth),
            max_tokens=4096,
        )

    def _node_detail(self, node: _Node, context: str) -> NodeDetailResult:
        return self.llm.complete(
            NodeDetailResult,
            system=SYSTEM_PRIMA_SEED,
            user=node_details_prompt(
                label=node.label,
                parent_label=node.parent_label or "root",
                context=context,
                members=node.members,
                children_labels=[c.label for c in node.children],
            ),
            max_tokens=2048,
        )

    def _dialectical_check(
        self,
        unplaced: list[str],
        all_nodes: list[_Node],
        leaf_nodes: list[_Node],
        context: str,
    ) -> list[UnresolvedValue]:
        from .schemas import PlacementResult

        tree_summary = _tree_summary(all_nodes)
        result = self.llm.complete(
            PlacementResult,
            system=SYSTEM_PRIMA_SEED,
            user=dialectical_check_prompt(unplaced, tree_summary),
            max_tokens=4096,
        )

        # Assign placed values to their target nodes
        cid_to_node = {n.canonical_id: n for n in all_nodes}
        for placed in result.placed:
            target = cid_to_node.get(placed.canonical_id)
            if target:
                if placed.normalized not in target.members:
                    target.members.append(placed.normalized)
            else:
                # Canonical ID not found — attach to best leaf as fallback
                if leaf_nodes:
                    leaf_nodes[0].members.append(placed.normalized)

        unresolved = [UnresolvedValue(raw=u.raw, reason=u.reason) for u in result.unresolved]
        if unresolved:
            console.print(f"  [yellow]{len(unresolved)} values remain unresolved after Q3[/]")
        return unresolved


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_child(b: BranchDef, parent: _Node) -> _Node:
    canonical_id = f"{parent.canonical_id}.{slugify(b.key)}"
    return _Node(
        key=b.key,
        label=b.label,
        canonical_id=canonical_id,
        parent_label=parent.label,
        members=list(b.members),
        criteria_includes=list(b.criteria_includes),
        criteria_excludes=list(b.criteria_excludes),
    )


def _collect_unplaced(all_normalized: list[str], leaf_nodes: list[_Node]) -> list[str]:
    placed = {m for n in leaf_nodes for m in n.members}
    return [v for v in all_normalized if v not in placed]


def _tree_summary(nodes: list[_Node]) -> str:
    lines = []
    for n in nodes:
        indent = "  " * n.canonical_id.count(".")
        lines.append(f"{indent}- {n.canonical_id}: {n.label}")
    return "\n".join(lines)


def _build_placement_map(
    nodes: list[_Node],
    raw_by_norm: dict[str, list[str]],
    def_by_norm: dict[str, str],
    seed_id: str,
    prefix: str,
) -> PlacementMap:
    by_cid: dict[str, list[PlacementEntity]] = defaultdict(list)

    for node in nodes:
        for member in node.members:
            entity = PlacementEntity(
                normalized=member,
                definition=def_by_norm.get(member),
                raw_samples=raw_by_norm.get(member, [member]),
            )
            by_cid[node.canonical_id].append(entity)

    placements = [
        Placement(canonical_id=cid, entities=entities)
        for cid, entities in sorted(by_cid.items())
    ]
    return PlacementMap(
        seed_id=seed_id,
        seed_version="1.0.0",
        generated=str(date.today()),
        placements=placements,
    )


def _build_seed(foundation: FoundationResult, root: _Node, all_nodes: list[_Node]) -> Seed:
    levels: dict[str, SeedLevel] = {}

    def add_levels(node: _Node, level_num: int) -> None:
        if not node.children:
            return
        level_key = f"L{level_num}" if node.parent_label is None else f"L{level_num}_{node.key}"
        levels[level_key] = SeedLevel(
            dimension=node.criteria_includes[0] if node.criteria_includes else node.label,
            question=node.question or f"What kind of {node.label.lower()} is this?",
            branches=[c.key for c in node.children],
        )
        for child in node.children:
            add_levels(child, level_num + 1)

    add_levels(root, 1)

    return Seed(
        seed_id=foundation.seed_id,
        description=foundation.description,
        context=SeedContext(
            Q0_answer=foundation.context_Q0,
            Q0b_answer=foundation.context_Q0b,
        ),
        normalization_rules={r: True for r in foundation.normalization_rules},
        levels=levels,
    )


def _build_output_nodes(internal: list[_Node]) -> list[Node]:
    from ..taxonomy.models import EdgeCase as EC, DecisionRecord as DR

    nodes = []
    for n in internal:
        nodes.append(
            Node(
                node=n.label,
                canonical_id=n.canonical_id,
                question=n.question or f"Does this belong to {n.label}?",
                criteria=NodeCriteria(
                    includes=n.criteria_includes,
                    excludes=n.criteria_excludes,
                ),
                parent=n.parent_label,
                children=[c.canonical_id for c in n.children],
                edge_cases=n.edge_cases,
                decision_record=n.decision_record,
            )
        )
    return nodes
