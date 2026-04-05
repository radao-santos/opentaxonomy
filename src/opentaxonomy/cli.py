import click

from .commands.create import run_create
from .commands.expand import run_expand
from .commands.run import run_run


def _parse_sheet(sheet: str | None) -> str | int | None:
    if sheet is None:
        return None
    return int(sheet) if sheet.isdigit() else sheet


_SHARED = [
    click.option("--input", "-i", "input_source", required=True,
                 help="Input file path (csv/tsv/json/xlsx/parquet) or database connection string"),
    click.option("--column", "-c", required=True,
                 help="Column containing the raw values to classify"),
    click.option("--api-key", envvar="ANTHROPIC_API_KEY", default=None,
                 help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"),
    click.option("--model", default="claude-sonnet-4-6", show_default=True,
                 help="Claude model to use"),
    click.option("--db-table", default=None,
                 help="Table name (required when input is a database connection string)"),
    click.option("--sheet", default=None,
                 help="Sheet name or index for Excel files (default: first sheet)"),
]


def shared_options(f):
    for option in reversed(_SHARED):
        f = option(f)
    return f


@click.group()
@click.version_option(package_name="opentaxonomy")
def main():
    """OpenTaxonomy: LLM-powered semantic taxonomy generator.\n
    \b
    Commands:
      create   Run the Prima Seed protocol to generate a new taxonomy from raw data.
      run      Place new/unseen values using an existing taxonomy seed.
    """


@main.command()
@shared_options
@click.option("--output-dir", "-o", default="./taxonomy", show_default=True,
              help="Directory to write seed.yaml, node files, and placement_map.yaml")
@click.option("--domain-hint", default="", show_default=False,
              help="Optional hint about the data domain (e.g. 'German grocery products')")
def create(input_source, column, output_dir, api_key, model, domain_hint, db_table, sheet):
    """Run Prima Seed: generate a full taxonomy from raw data.

    \b
    Outputs (written to --output-dir):
      seed.yaml           Domain seed capturing context and tree structure
      nodes/*.yaml        One ontological contract per tree node
      placement_map.yaml  Raw values mapped to canonical IDs (source of truth)

    The input file/table is enriched with two new columns:
      {column}_normalized   Cleaned entity name
      canonical_id          Taxonomy path (e.g. ft.expenditure.variable.groceries)
    """
    if not api_key:
        raise click.UsageError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY or use --api-key."
        )
    run_create(
        input_source=input_source,
        column=column,
        output_dir=output_dir,
        api_key=api_key,
        model=model,
        domain_hint=domain_hint,
        db_table=db_table,
        sheet=_parse_sheet(sheet),
    )


@main.command()
@shared_options
@click.option("--seed-dir", "-s", default="./taxonomy", show_default=True,
              help="Directory containing seed.yaml, nodes/, and placement_map.yaml")
def run(input_source, column, seed_dir, api_key, model, db_table, sheet):
    """Place new values using an existing taxonomy seed.

    \b
    Only processes values not already in placement_map.yaml.
    Appends new placements to placement_map.yaml (source of truth).
    Writes {column}_normalized and canonical_id columns back to the source.
    """
    if not api_key:
        raise click.UsageError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY or use --api-key."
        )
    run_run(
        input_source=input_source,
        column=column,
        seed_dir=seed_dir,
        api_key=api_key,
        model=model,
        db_table=db_table,
        sheet=_parse_sheet(sheet),
    )


@main.command()
@click.option("--seed-dir", "-s", default="./taxonomy", show_default=True,
              help="Directory containing seed.yaml, nodes/, and placement_map.yaml")
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", default=None,
              help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
@click.option("--model", default="claude-sonnet-4-6", show_default=True,
              help="Claude model to use")
@click.option("--max-retries", default=2, show_default=True,
              help="Max graft attempts before routing to the Other bucket")
def expand(seed_dir, api_key, model, max_retries):
    """Graft new branches to cover unresolved values.

    \b
    Reads unresolved values from placement_map.yaml, then:
      1. Traverses the existing tree to find where each value fits
      2. Creates new branches at the right level (vertical or horizontal growth)
      3. After --max-retries failed attempts, routes remaining to {prefix}.other
    Updates seed.yaml, nodes/, and placement_map.yaml in place.
    """
    if not api_key:
        raise click.UsageError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY or use --api-key."
        )
    run_expand(
        seed_dir=seed_dir,
        api_key=api_key,
        model=model,
        max_retries=max_retries,
    )
