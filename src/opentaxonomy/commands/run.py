from pathlib import Path

import click
from rich.console import Console

from ..io.db_sources import SQLSource
from ..io.file_sources import source_from_path
from ..llm.client import TaxonomyLLM
from ..llm.run_flow import RunFlow
from ..taxonomy.writer import write_placement_map

console = Console()

_DB_PREFIXES = ("postgresql://", "postgresql+", "mysql://", "mysql+", "sqlite:///", "mssql+")


def run_run(
    input_source: str,
    column: str,
    seed_dir: str,
    api_key: str,
    model: str,
    db_table: str | None,
    sheet: str | int | None,
) -> None:
    seed_path = Path(seed_dir) / "seed.yaml"
    pm_path = Path(seed_dir) / "placement_map.yaml"
    nodes_dir = Path(seed_dir) / "nodes"

    if not seed_path.exists():
        raise click.UsageError(f"seed.yaml not found in '{seed_dir}'")
    if not pm_path.exists():
        raise click.UsageError(f"placement_map.yaml not found in '{seed_dir}'")

    # Load data
    console.print(f"\n[bold]Input:[/] {input_source}")
    source = _make_source(input_source, db_table, sheet)
    df = source.read()

    if column not in df.columns:
        raise click.UsageError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )

    raw_values = df[column].dropna().astype(str).tolist()
    unique_count = len(set(raw_values))
    console.print(f"  {len(raw_values)} rows, {unique_count} unique values in '[bold]{column}[/]'")

    # Run placement flow
    llm = TaxonomyLLM(api_key=api_key, model=model)
    flow = RunFlow(llm)

    console.rule("[bold cyan]Placement Flow[/]")
    updated_pm, raw_to_normalized, raw_to_canonical = flow.run(
        raw_values, pm_path, seed_path, nodes_dir
    )

    # Write updated placement map (source of truth)
    write_placement_map(updated_pm, Path(seed_dir))
    console.print(f"[green]✓[/] Placement map updated: {pm_path}")

    # Enrich dataframe and write back
    console.rule("[bold]Writing enriched data back to source[/]")
    df[f"{column}_normalized"] = df[column].map(raw_to_normalized)
    df["canonical_id"] = df[column].map(raw_to_canonical)
    source.write(df)
    console.print(f"  [green]✓[/] Added columns: [italic]{column}_normalized[/], [italic]canonical_id[/]")

    n_unresolved = len(updated_pm.unresolved)
    console.rule()
    console.print(f"[bold green]Done.[/]  Unresolved: {n_unresolved}")


def _make_source(input_source: str, db_table: str | None, sheet: str | int | None):
    if any(input_source.startswith(p) for p in _DB_PREFIXES):
        if not db_table:
            raise click.UsageError("--db-table is required for database sources")
        return SQLSource(input_source, db_table)
    kwargs = {}
    if sheet is not None:
        kwargs["sheet_name"] = sheet
    return source_from_path(input_source, **kwargs)
