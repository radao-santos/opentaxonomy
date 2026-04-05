"""
ExpandFlow — incremental tree growth for unresolved values.

Algorithm:
  1. Load unresolved values from placement_map.yaml
  2. Traverse tree top-down: batch-ask which child each value fits at each level
     - Fits a child → follow it deeper
     - Fits no child → graft point found here
  3. At each graft point, run localized branch generation
  4. Write new nodes, update parent children lists, rebuild seed levels
  5. Repeat up to max_retries for any values still unplaced after grafting
  6. After max_retries: remaining values → {prefix}.other (the catch-all bucket)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
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
    SeedLevel,
    UnresolvedValue,
)
from ..utils.canonical_id import content_hash, safe_child_id, slugify
from .client import TaxonomyLLM
from .prompts import (
    SYSTEM_PRIMA_SEED,
    graft_prompt,
    tree_fit_prompt,
)
from .schemas import GraftResult, TreeFitResult

console = Console()

_OTHER_KEY = "other"


# ── Tree loader ───────────────────────────────────────────────────────────────

def load_tree(nodes_dir: Path) -> dict[str, Node]:
    """Load all node YAML files into a {canonical_id: Node} dict."""
    tree: dict[str, Node] = {}
    if not nodes_dir.exists():
        return tree
    for f in sorted(nodes_dir.glob("*.yaml")):
        with open(f, encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
        if data:
            node = Node.model_validate(data)
            tree[node.canonical_id] = node
    return tree


def get_root(tree: dict[str, Node]) -> Optional[Node]:
    for node in tree.values():
        if node.parent is None:
            return node
    return None


def get_prefix(tree: dict[str, Node]) -> str:
    root = get_root(tree)
    return root.canonical_id if root else "other"


# ── Main flow ─────────────────────────────────────────────────────────────────

class ExpandFlow:
    def __init__(self, llm: TaxonomyLLM, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

    def run(
        self,
        seed_path: Path,
        nodes_dir: Path,
        pm_path: Path,
    ) -> tuple[list[Node], int, int]:
        """
        Expand the taxonomy to cover unresolved values.

        Returns:
          - list of new Node objects created
          - count of values newly placed
          - count of values sent to Other (after max_retries)
        """
        # Load state
        with open(pm_path, encoding="utf-8") as f:
            pm_data = yaml.safe_load(f)
        pm = PlacementMap.model_validate(pm_data)

        with open(seed_path, encoding="utf-8") as f:
            seed_data = yaml.safe_load(f)
        seed = Seed.model_validate(seed_data)

        tree = load_tree(nodes_dir)
        if not tree:
            console.print("[red]No node files found. Run 'create' first.[/]")
            return [], 0, 0

        prefix = get_prefix(tree)
        existing_ids = set(tree.keys())

        unresolved = [u.raw for u in pm.unresolved]
        if not unresolved:
            console.print("[green]No unresolved values in placement map.[/]")
            return [], 0, 0

        console.print(f"[bold blue]Expand:[/] {len(unresolved)} unresolved values")

        # Load context from seed
        context = seed.context.Q0b_answer

        all_new_nodes: list[Node] = []
        newly_placed: dict[str, str] = {}  # normalized → canonical_id
        remaining = list(unresolved)

        for attempt in range(1, self.max_retries + 1):
            if not remaining:
                break
            console.print(f"\n[bold]Attempt {attempt}/{self.max_retries}[/] — {len(remaining)} values")

            # Find graft points via top-down tree traversal
            graft_points = self._find_graft_points(remaining, tree, context, prefix)

            # Generate new branches at each graft point
            attempt_placed: list[str] = []
            attempt_unplaced: list[str] = []

            for parent_id, values_here in graft_points.items():
                parent = tree.get(parent_id)
                if not parent:
                    attempt_unplaced.extend(values_here)
                    continue

                console.print(f"  Grafting {len(values_here)} values under [italic]{parent_id}[/]")
                new_nodes, placed, unplaced = self._graft(
                    values_here, parent, tree, existing_ids, context
                )

                # Register new nodes
                for node in new_nodes:
                    tree[node.canonical_id] = node
                    existing_ids.add(node.canonical_id)
                    all_new_nodes.append(node)
                    # Update parent's children list
                    if node.canonical_id not in tree[parent_id].children:
                        tree[parent_id].children.append(node.canonical_id)

                # Update placement map
                self._update_pm(pm, placed, tree)

                for norm, cid in placed.items():
                    newly_placed[norm] = cid
                    attempt_placed.append(norm)
                attempt_unplaced.extend(unplaced)

            console.print(f"  Placed: {len(attempt_placed)}  Still unplaced: {len(attempt_unplaced)}")
            remaining = attempt_unplaced

        # After max_retries: route remaining to {prefix}.other
        count_other = 0
        if remaining:
            console.print(f"\n[yellow]Max retries reached. Routing {len(remaining)} values to Other bucket.[/]")
            other_id = f"{prefix}.{_OTHER_KEY}"
            other_node = self._ensure_other_node(other_id, prefix, tree, nodes_dir)
            if other_node and other_id not in [n.canonical_id for n in all_new_nodes]:
                all_new_nodes.append(other_node)

            for value in remaining:
                entity = PlacementEntity(
                    normalized=value,
                    definition="Value that could not be classified into any established category.",
                    raw_samples=[value],
                )
                self._add_to_pm(pm, other_id, entity)
                newly_placed[value] = other_id
                count_other += 1

            # Update root's children to include Other
            root = get_root(tree)
            if root and other_id not in root.children:
                root.children.append(other_id)
                tree[root.canonical_id] = root

        # Remove placed values from pm.unresolved
        placed_set = set(newly_placed.keys())
        pm.unresolved = [u for u in pm.unresolved if u.raw not in placed_set]

        # Write all updated files
        self._write_results(all_new_nodes, tree, pm, pm_path, nodes_dir, seed, seed_path, prefix)

        return all_new_nodes, len(newly_placed) - count_other, count_other

    # ── Tree traversal ────────────────────────────────────────────────────────

    def _find_graft_points(
        self,
        values: list[str],
        tree: dict[str, Node],
        context: str,
        prefix: str,
    ) -> dict[str, list[str]]:
        """
        Traverse tree top-down, batch-routing values at each level.
        Returns {parent_canonical_id: [values to graft here]}.
        """
        root = get_root(tree)
        if not root:
            return {prefix: values}

        # pending[node_id] = list of values being routed through that node
        pending: dict[str, list[str]] = {root.canonical_id: list(values)}
        graft_points: dict[str, list[str]] = defaultdict(list)

        while pending:
            next_pending: dict[str, list[str]] = defaultdict(list)

            for node_id, node_values in pending.items():
                node = tree.get(node_id)
                if not node:
                    graft_points[node_id].extend(node_values)
                    continue

                # Get children (excluding the Other bucket)
                children = [
                    tree[cid] for cid in node.children
                    if cid in tree and not cid.endswith(f".{_OTHER_KEY}")
                ]

                if not children:
                    # Leaf node — values graft here as new siblings under the parent
                    parent_id = node.parent or node.canonical_id
                    graft_points[parent_id].extend(node_values)
                    continue

                # Ask LLM: which child does each value fit?
                children_info = [
                    {
                        "canonical_id": c.canonical_id,
                        "label": c.node,
                        "question": c.question,
                        "includes": c.criteria.includes,
                    }
                    for c in children
                ]
                fit_result = self.llm.complete(
                    TreeFitResult,
                    system=SYSTEM_PRIMA_SEED,
                    user=tree_fit_prompt(node_values, node.node, node.canonical_id, context, children_info),
                    max_tokens=4096,
                )

                for fit in fit_result.fits:
                    if fit.matching_canonical_id and fit.matching_canonical_id in tree:
                        next_pending[fit.matching_canonical_id].append(fit.value)
                    else:
                        graft_points[node_id].append(fit.value)

            pending = dict(next_pending)

        return dict(graft_points)

    # ── Branch generation ─────────────────────────────────────────────────────

    def _graft(
        self,
        values: list[str],
        parent: Node,
        tree: dict[str, Node],
        existing_ids: set[str],
        context: str,
    ) -> tuple[list[Node], dict[str, str], list[str]]:
        """
        Generate new child branches for values that don't fit existing children.
        Returns: (new_nodes, {normalized: canonical_id}, still_unplaced)
        """
        result = self.llm.complete(
            GraftResult,
            system=SYSTEM_PRIMA_SEED,
            user=graft_prompt(
                values=values,
                parent_label=parent.node,
                parent_question=parent.question,
                parent_includes=parent.criteria.includes,
                context=context,
            ),
            max_tokens=4096,
        )

        new_nodes: list[Node] = []
        placed: dict[str, str] = {}

        for branch in result.branches:
            cid = safe_child_id(branch.key, parent.canonical_id, existing_ids)
            existing_ids.add(cid)

            node = Node(
                node=branch.label,
                canonical_id=cid,
                question=branch.question,
                criteria=NodeCriteria(
                    includes=branch.criteria_includes,
                    excludes=branch.criteria_excludes,
                ),
                parent=parent.canonical_id,
                children=[],
                edge_cases=branch.edge_cases,
                content_hash=content_hash(branch.question, branch.criteria_includes),
            )
            new_nodes.append(node)

            for member in branch.members:
                placed[member] = cid

        return new_nodes, placed, list(result.unplaced)

    # ── Placement map helpers ─────────────────────────────────────────────────

    def _update_pm(self, pm: PlacementMap, placed: dict[str, str], tree: dict[str, Node]) -> None:
        cid_index = {p.canonical_id: p for p in pm.placements}
        by_cid: dict[str, list[str]] = defaultdict(list)
        for norm, cid in placed.items():
            by_cid[cid].append(norm)

        for cid, members in by_cid.items():
            node = tree.get(cid)
            for member in members:
                entity = PlacementEntity(
                    normalized=member,
                    raw_samples=[member],
                )
                if cid in cid_index:
                    cid_index[cid].entities.append(entity)
                else:
                    new_placement = Placement(canonical_id=cid, entities=[entity])
                    pm.placements.append(new_placement)
                    cid_index[cid] = new_placement

    def _add_to_pm(self, pm: PlacementMap, cid: str, entity: PlacementEntity) -> None:
        for placement in pm.placements:
            if placement.canonical_id == cid:
                placement.entities.append(entity)
                return
        pm.placements.append(Placement(canonical_id=cid, entities=[entity]))

    # ── Other bucket ──────────────────────────────────────────────────────────

    def _ensure_other_node(
        self,
        other_id: str,
        prefix: str,
        tree: dict[str, Node],
        nodes_dir: Path,
    ) -> Optional[Node]:
        if other_id in tree:
            return None  # already exists

        root = get_root(tree)
        parent_label = root.node if root else "root"
        node = Node(
            node="Other",
            canonical_id=other_id,
            question="Does this value not fit any established category in this taxonomy?",
            criteria=NodeCriteria(
                includes=[
                    "Values that could not be classified after multiple attempts",
                    "Entities whose nature is unclear or ambiguous",
                ],
                excludes=[
                    "Any value that fits an existing node's criteria",
                ],
            ),
            parent=prefix,
            children=[],
            content_hash=content_hash(
                "Does this value not fit any established category?",
                ["Values that could not be classified after multiple attempts"],
            ),
        )
        tree[other_id] = node
        return node

    # ── File writing ──────────────────────────────────────────────────────────

    def _write_results(
        self,
        new_nodes: list[Node],
        tree: dict[str, Node],
        pm: PlacementMap,
        pm_path: Path,
        nodes_dir: Path,
        seed: Seed,
        seed_path: Path,
        prefix: str,
    ) -> None:
        from ..taxonomy.writer import write_node, write_placement_map, write_seed

        # Write new node files
        output_dir = nodes_dir.parent
        for node in new_nodes:
            path = write_node(node, output_dir)
            console.print(f"  [green]+[/] {path}")

        # Update existing node files that had children added (parents of new nodes)
        parent_ids = {n.parent for n in new_nodes if n.parent}
        for pid in parent_ids:
            parent = tree.get(pid)
            if parent:
                path = write_node(parent, output_dir)
                console.print(f"  [blue]~[/] {path} (children updated)")

        # Rebuild seed levels from expanded tree
        root = get_root(tree)
        if root:
            seed.levels = _rebuild_levels(tree, root.canonical_id)
        write_seed(seed, output_dir)

        # Write updated placement map
        write_placement_map(pm, output_dir)


# ── Seed level rebuild ────────────────────────────────────────────────────────

def _rebuild_levels(tree: dict[str, Node], root_id: str) -> dict[str, SeedLevel]:
    levels: dict[str, SeedLevel] = {}

    def walk(node_id: str, depth: int) -> None:
        node = tree.get(node_id)
        if not node or not node.children:
            return

        level_key = f"L{depth}" if depth == 1 else f"L{depth}_{node.canonical_id.split('.')[-1]}"
        levels[level_key] = SeedLevel(
            dimension=node.criteria.includes[0] if node.criteria.includes else node.node,
            question=node.question,
            branches=[cid.split(".")[-1] for cid in node.children],
        )
        for child_id in node.children:
            walk(child_id, depth + 1)

    walk(root_id, 1)
    return levels
