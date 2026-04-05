from pathlib import Path

import click
from rich.console import Console

from ..llm.client import TaxonomyLLM
from ..llm.expand_flow import ExpandFlow

console = Console()


def run_expand(
    seed_dir: str,
    api_key: str,
    model: str,
    max_retries: int,
) -> None:
    seed_path = Path(seed_dir) / "seed.yaml"
    pm_path = Path(seed_dir) / "placement_map.yaml"
    nodes_dir = Path(seed_dir) / "nodes"

    if not seed_path.exists():
        raise click.UsageError(f"seed.yaml not found in '{seed_dir}'")
    if not pm_path.exists():
        raise click.UsageError(f"placement_map.yaml not found in '{seed_dir}'")

    llm = TaxonomyLLM(api_key=api_key, model=model)
    flow = ExpandFlow(llm, max_retries=max_retries)

    console.rule("[bold cyan]Expand Flow[/]")
    new_nodes, n_placed, n_other = flow.run(seed_path, nodes_dir, pm_path)

    console.rule()
    console.print(f"[bold green]Done.[/]")
    console.print(f"  New nodes created : {len(new_nodes)}")
    console.print(f"  Values placed     : {n_placed}")
    console.print(f"  Routed to Other   : {n_other}")
