"""
test command — iterative sample-build-validate loop.

Flow:
  1. Sample `sample_size` unique values from input data
  2. Run CreateFlow on sample → build a test taxonomy in `output_dir`
  3. Run the placement agent loop on `validate_size` held-out values:
       each batch of 50 spawns one placement agent call (RunFlow)
  4. Print a quality report: placement rate, category breakdown, unresolved list
"""
from __future__ import annotations

import random
import tempfile
from collections import Counter
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..io.db_sources import SQLSource
from ..io.file_sources import source_from_path
from ..llm.client import TaxonomyLLM
from ..llm.create_flow import CreateFlow
from ..llm.run_flow import RunFlow
from ..taxonomy.writer import write_node, write_placement_map, write_seed

console = Console()

_DB_PREFIXES = ("postgresql://", "postgresql+", "mysql://", "mysql+", "sqlite:///", "mssql+")


def run_test(
    input_source: str,
    column: str,
    output_dir: str,
    api_key: str,
    model: str,
    domain_hint: str,
    db_table: str | None,
    sheet: str | int | None,
    sample_size: int,
    validate_size: int,
    seed_value: int,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    console.print(f"\n[bold]Input:[/] {input_source}")
    source = _make_source(input_source, db_table, sheet)
    df = source.read()

    if column not in df.columns:
        raise click.UsageError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )

    all_unique = df[column].dropna().astype(str).unique().tolist()
    console.print(f"  {len(all_unique)} unique values in '[bold]{column}[/]'")

    if sample_size >= len(all_unique):
        raise click.UsageError(
            f"--sample-size {sample_size} ≥ total unique values ({len(all_unique)}). "
            "Reduce --sample-size or use 'create' instead."
        )

    # ── 2. Split: sample for building, rest for validation ────────────────────
    rng = random.Random(seed_value)
    shuffled = list(all_unique)
    rng.shuffle(shuffled)

    sample_values = shuffled[:sample_size]
    held_out_pool = shuffled[sample_size:]

    if validate_size == 0 or validate_size >= len(held_out_pool):
        validate_values = held_out_pool
    else:
        validate_values = held_out_pool[:validate_size]

    console.rule("[bold cyan]Step 1 — Build taxonomy from sample[/]")
    console.print(
        f"  [bold]{sample_size}[/] values used to build  |  "
        f"[bold]{len(validate_values)}[/] held-out values for validation"
    )

    # ── 3. Run CreateFlow on sample ───────────────────────────────────────────
    llm_create = TaxonomyLLM(api_key=api_key, model=model)
    flow = CreateFlow(llm_create)
    seed, nodes, placement_map = flow.run(sample_values, domain_hint=domain_hint)

    seed_path = write_seed(seed, output_path)
    for node in nodes:
        write_node(node, output_path)
    pm_path = write_placement_map(placement_map, output_path)

    n_sample_placed = sum(len(p.entities) for p in placement_map.placements)
    console.print(
        f"  [green]✓[/] Taxonomy written to [italic]{output_path}[/]\n"
        f"  {len(nodes)} nodes  |  {n_sample_placed} sample entities placed  |  "
        f"{len(placement_map.unresolved)} unresolved in sample"
    )

    # ── 4. Agent spawning loop — place held-out values ────────────────────────
    console.rule("[bold cyan]Step 2 — Agent placement loop (held-out validation)[/]")

    nodes_dir = output_path / "nodes"
    llm_run = TaxonomyLLM(api_key=api_key, model="claude-haiku-4-5-20251001")
    run_flow = RunFlow(llm_run)

    _BATCH = 50
    total_batches = (len(validate_values) + _BATCH - 1) // _BATCH
    all_raw_to_canonical: dict[str, str] = {}

    for batch_idx in range(total_batches):
        batch = validate_values[batch_idx * _BATCH : (batch_idx + 1) * _BATCH]
        console.print(
            f"\n  [bold blue]Placement Agent {batch_idx + 1}/{total_batches}[/]"
            f"  ({len(batch)} values)"
        )
        updated_pm, _, raw_to_cid = run_flow.run(batch, pm_path, seed_path, nodes_dir)
        all_raw_to_canonical.update(raw_to_cid)
        # Persist updated placement map so next agent sees all placements so far
        write_placement_map(updated_pm, output_path)

    # ── 5. Quality report ─────────────────────────────────────────────────────
    console.rule("[bold]Quality Report[/]")

    # Reload final placement map for stats
    import yaml
    from ..taxonomy.models import PlacementMap
    with open(pm_path, encoding="utf-8") as f:
        final_pm = PlacementMap.model_validate(yaml.safe_load(f))

    # Categorise validation values
    unresolved_raw = {u.raw for u in final_pm.unresolved}
    placed_count = 0
    unresolved_count = 0
    cid_counter: Counter = Counter()

    for raw in validate_values:
        cid = all_raw_to_canonical.get(raw)
        if cid:
            placed_count += 1
            cid_counter[cid] += 1
        else:
            unresolved_count += 1

    total_v = len(validate_values)
    placement_rate = placed_count / total_v * 100 if total_v else 0

    console.print(
        f"\n  Validation values:  [bold]{total_v}[/]\n"
        f"  Placed:             [bold green]{placed_count}[/]  "
        f"({placement_rate:.1f}%)\n"
        f"  Unresolved:         [bold yellow]{unresolved_count}[/]  "
        f"({100 - placement_rate:.1f}%)"
    )

    # Category distribution table
    if cid_counter:
        table = Table(title="Category Distribution (validation set)", show_lines=False)
        table.add_column("Canonical ID", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Share", justify="right")
        for cid, count in cid_counter.most_common(20):
            share = count / placed_count * 100
            table.add_row(cid, str(count), f"{share:.1f}%")
        if len(cid_counter) > 20:
            table.add_row("…", f"(+{len(cid_counter) - 20} more)", "")
        console.print(table)

    # Unresolved sample
    unresolved_in_validate = [v for v in validate_values if v not in all_raw_to_canonical]
    if unresolved_in_validate:
        console.print(f"\n  [yellow]Unresolved sample (up to 10):[/]")
        for v in unresolved_in_validate[:10]:
            console.print(f"    – {v}")

    console.rule()
    console.print(
        f"[bold green]Done.[/]  Taxonomy saved to [italic]{output_path}[/]\n"
        f"  Run '[bold]opentaxonomy expand --seed-dir {output_path}[/]' to graft "
        f"branches for unresolved values."
    )


def _make_source(input_source: str, db_table: str | None, sheet: str | int | None):
    if any(input_source.startswith(p) for p in _DB_PREFIXES):
        if not db_table:
            raise click.UsageError("--db-table is required for database sources")
        return SQLSource(input_source, db_table)
    kwargs = {}
    if sheet is not None:
        kwargs["sheet_name"] = sheet
    return source_from_path(input_source, **kwargs)
