"""
RunFlow — places new/unseen values into an existing taxonomy.
  1. Load existing placement_map.yaml
  2. Normalize incoming raw values
  3. Find values not yet in the placement map
  4. Place new normalized values against the existing tree (seed.yaml + nodes/)
  5. Append new placements to placement_map.yaml (source of truth)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml
from rich.console import Console

from ..taxonomy.models import Placement, PlacementEntity, PlacementMap, UnresolvedValue
from .client import TaxonomyLLM
from .prompts import SYSTEM_PLACER, SYSTEM_PRIMA_SEED, normalize_prompt, placement_prompt
from .schemas import NormalizationResult, NormalizedEntity, PlacementResult

console = Console()

_NORMALIZE_BATCH = 150
_PLACEMENT_BATCH = 50


class RunFlow:
    def __init__(self, llm: TaxonomyLLM):
        self.llm = llm

    def run(
        self,
        raw_values: list[str],
        placement_map_path: Path,
        seed_path: Path,
        nodes_dir: Path,
    ) -> tuple[PlacementMap, dict[str, str], dict[str, str]]:
        """
        Returns:
          - updated PlacementMap
          - raw_to_normalized  {raw: normalized}
          - raw_to_canonical   {raw: canonical_id}
        """
        # Load existing placement map
        with open(placement_map_path, encoding="utf-8") as f:
            pm_data = yaml.safe_load(f)
        pm = PlacementMap.model_validate(pm_data)

        # Build lookup tables from existing map
        raw_to_normalized: dict[str, str] = {}
        raw_to_canonical: dict[str, str] = {}
        known_normalized: set[str] = set()

        for placement in pm.placements:
            for entity in placement.entities:
                known_normalized.add(entity.normalized)
                for raw in entity.raw_samples:
                    raw_to_normalized[raw] = entity.normalized
                    raw_to_canonical[raw] = placement.canonical_id

        # Find raw values not yet mapped
        new_raw = [v for v in raw_values if v not in raw_to_canonical]

        if not new_raw:
            console.print("[green]No new values to place.[/]")
            return pm, raw_to_normalized, raw_to_canonical

        unique_new_raw = list(dict.fromkeys(new_raw))
        console.print(f"[bold blue]Normalize:[/] {len(unique_new_raw)} new raw values…")

        # Normalize new raw values
        entities = self._normalize(unique_new_raw)
        for e in entities:
            for raw in e.raw_samples:
                raw_to_normalized[raw] = e.normalized

        # Split into truly new normalized entities vs already-known ones
        new_entities = [e for e in entities if e.normalized not in known_normalized]
        known_entities = [e for e in entities if e.normalized in known_normalized]

        # For already-known entities, append any new raw_samples to existing placements
        if known_entities:
            self._append_raw_samples(pm, known_entities, raw_to_canonical, raw_to_normalized)

        if not new_entities:
            console.print("[green]All new values normalize to already-known entities.[/]")
            return pm, raw_to_normalized, raw_to_canonical

        console.print(f"[bold blue]Place:[/] {len(new_entities)} new normalized entities…")

        # Build tree context for placement
        tree_yaml = self._build_tree_context(seed_path, nodes_dir)

        # Place new normalized values in batches
        new_normalized = [e.normalized for e in new_entities]
        raw_by_norm = {e.normalized: e.raw_samples for e in new_entities}
        def_by_norm = {e.normalized: e.definition for e in new_entities}

        all_placed: list = []
        all_unresolved: list = []

        for i in range(0, len(new_normalized), _PLACEMENT_BATCH):
            batch = new_normalized[i : i + _PLACEMENT_BATCH]
            result = self.llm.complete(
                PlacementResult,
                system=SYSTEM_PLACER,
                user=placement_prompt(batch, tree_yaml),
                max_tokens=4096,
            )
            all_placed.extend(result.placed)
            all_unresolved.extend(result.unresolved)

        # Update placement map with new placements
        by_cid: dict[str, list[PlacementEntity]] = defaultdict(list)
        for placed in all_placed:
            entity = PlacementEntity(
                normalized=placed.normalized,
                definition=def_by_norm.get(placed.normalized),
                raw_samples=raw_by_norm.get(placed.normalized, [placed.normalized]),
            )
            by_cid[placed.canonical_id].append(entity)
            # Update lookup tables
            for raw in entity.raw_samples:
                raw_to_canonical[raw] = placed.canonical_id

        cid_index = {p.canonical_id: p for p in pm.placements}
        for cid, new_entities_list in by_cid.items():
            if cid in cid_index:
                cid_index[cid].entities.extend(new_entities_list)
            else:
                pm.placements.append(Placement(canonical_id=cid, entities=new_entities_list))

        for u in all_unresolved:
            pm.unresolved.append(UnresolvedValue(raw=u.raw, reason=u.reason))

        console.print(
            f"  [green]✓[/] {len(all_placed)} placed  "
            f"[yellow]{len(all_unresolved)} unresolved[/]"
        )
        return pm, raw_to_normalized, raw_to_canonical

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalize(self, raw_values: list[str]) -> list[NormalizedEntity]:
        from collections import defaultdict as dd
        merged: dict[str, list[str]] = dd(list)
        for i in range(0, len(raw_values), _NORMALIZE_BATCH):
            batch = raw_values[i : i + _NORMALIZE_BATCH]
            result = self.llm.complete(
                NormalizationResult,
                system=SYSTEM_PRIMA_SEED,
                user=normalize_prompt(batch),
                max_tokens=8096,
            )
            for e in result.entities:
                merged[e.normalized].extend(e.raw_samples)
        return [NormalizedEntity(normalized=n, raw_samples=list(dict.fromkeys(r)))
                for n, r in merged.items()]

    def _append_raw_samples(
        self,
        pm: PlacementMap,
        known_entities: list[NormalizedEntity],
        raw_to_canonical: dict[str, str],
        raw_to_normalized: dict[str, str],
    ) -> None:
        cid_entity_index: dict[tuple[str, str], PlacementEntity] = {}
        for placement in pm.placements:
            for entity in placement.entities:
                cid_entity_index[(placement.canonical_id, entity.normalized)] = entity

        for e in known_entities:
            # Find which canonical_id this normalized entity belongs to
            for placement in pm.placements:
                for entity in placement.entities:
                    if entity.normalized == e.normalized:
                        for raw in e.raw_samples:
                            if raw not in entity.raw_samples:
                                entity.raw_samples.append(raw)
                            raw_to_canonical[raw] = placement.canonical_id
                        break

    def _build_tree_context(self, seed_path: Path, nodes_dir: Path) -> str:
        parts = []
        with open(seed_path, encoding="utf-8") as f:
            parts.append(f"# SEED\n{f.read()}")
        if nodes_dir.exists():
            for node_file in sorted(nodes_dir.glob("*.yaml")):
                with open(node_file, encoding="utf-8") as f:
                    parts.append(f"# NODE: {node_file.stem}\n{f.read()}")
        return "\n\n".join(parts)
