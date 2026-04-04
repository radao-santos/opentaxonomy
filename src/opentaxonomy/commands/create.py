from pathlib import Path

import click
from rich.console import Console

from ..io.db_sources import SQLSource
from ..io.file_sources import source_from_path
from ..llm.client import TaxonomyLLM
from ..llm.create_flow import CreateFlow
from ..taxonomy.writer import write_node, write_placement_map, write_seed

console = Console()

_DB_PREFIXES = ("postgresql://", "postgresql+", "mysql://", "mysql+", "sqlite:///", "mssql+")


def run_create(
    input_source: str,
    column: str,
    output_dir: str,
    api_key: str,
    model: str,
    domain_hint: str,
    db_table: str | None,
    sheet: str | int | None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    console.print(f"\n[bold]Input:[/] {input_source}")
    source = _make_source(input_source, db_table, sheet)
    df = source.read()

    if column not in df.columns:
        raise click.UsageError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )

    raw_values = df[column].dropna().astype(str).unique().tolist()
    console.print(f"  {len(raw_values)} unique values in column '[bold]{column}[/]'")

    # Run Prima Seed
    llm = TaxonomyLLM(api_key=api_key, model=model)
    flow = CreateFlow(llm)

    console.rule("[bold cyan]Prima Seed Protocol[/]")
    seed, nodes, placement_map = flow.run(raw_values, domain_hint=domain_hint)

    # Write taxonomy files
    console.rule("[bold]Writing taxonomy files[/]")
    seed_path = write_seed(seed, output_path)
    console.print(f"  [green]✓[/] {seed_path}")

    for node in nodes:
        node_path = write_node(node, output_path)
        console.print(f"  [green]✓[/] {node_path}")

    pm_path = write_placement_map(placement_map, output_path)
    console.print(f"  [green]✓[/] {pm_path}")

    # Enrich dataframe and write back
    console.rule("[bold]Writing enriched data back to source[/]")
    raw_to_norm, raw_to_cid = _build_lookups(placement_map)
    df[f"{column}_normalized"] = df[column].map(raw_to_norm)
    df["canonical_id"] = df[column].map(raw_to_cid)
    source.write(df)
    console.print(f"  [green]✓[/] Added columns: [italic]{column}_normalized[/], [italic]canonical_id[/]")

    # Summary
    n_placed = sum(len(p.entities) for p in placement_map.placements)
    n_unresolved = len(placement_map.unresolved)
    console.rule()
    console.print(f"[bold green]Done.[/]  Taxonomy → [italic]{output_path}[/]")
    console.print(f"  Nodes: {len(nodes)}  |  Placed entities: {n_placed}  |  Unresolved: {n_unresolved}")


def _make_source(input_source: str, db_table: str | None, sheet: str | int | None):
    if any(input_source.startswith(p) for p in _DB_PREFIXES):
        if not db_table:
            raise click.UsageError("--db-table is required for database sources")
        return SQLSource(input_source, db_table)
    kwargs = {}
    if sheet is not None:
        kwargs["sheet_name"] = sheet
    return source_from_path(input_source, **kwargs)


def _build_lookups(pm) -> tuple[dict, dict]:
    raw_to_norm: dict[str, str] = {}
    raw_to_cid: dict[str, str] = {}
    for placement in pm.placements:
        for entity in placement.entities:
            for raw in entity.raw_samples:
                raw_to_norm[raw] = entity.normalized
                raw_to_cid[raw] = placement.canonical_id
    return raw_to_norm, raw_to_cid
