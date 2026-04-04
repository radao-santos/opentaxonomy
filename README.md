# OpenTaxonomy

LLM-powered semantic taxonomy generator for raw categorical data.

## What it does

OpenTaxonomy takes a column of messy raw values (bank transactions, product names, survey responses — anything) and generates a structured semantic taxonomy from it using Claude as the reasoning engine.

**Output for each value:**
- `{column}_normalized` — cleaned entity name (e.g. `"REWE SAGT DANKE 46654184/..."` → `"REWE"`)
- `canonical_id` — taxonomy path (e.g. `ft.expenditure.variable.groceries`)

The taxonomy itself is a set of YAML files — a `seed.yaml` capturing the domain structure, one `node.yaml` per tree node (each an ontological contract with inclusion/exclusion criteria and a decision record), and a `placement_map.yaml` that is the source of truth for all mappings.

## Architecture

The core is the **Prima Seed** — a universal questioning protocol that generates domain-specific taxonomic trees from any categorical data:

- **Q0** Identify the Form: what unifies all values?
- **Q0b** Establish context: what operational realm governs classification?
- **Q1** Primary differentiation: the most essential splitting criterion
- **Q2** Recursive differentiation: applied per branch at each level
- **Q3** Dialectical check: values that resist placement expose flawed criteria

## Installation

```bash
pip install opentaxonomy
```

Requires Python 3.11+ and an [Anthropic API key](https://console.anthropic.com/).

## Usage

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Generate a new taxonomy from raw data
opentaxonomy create -i transactions.csv -c description -o ./taxonomy

# Place new/unseen values into an existing taxonomy
opentaxonomy run -i new_data.csv -c description -s ./taxonomy
```

### Supported input formats

| Format | Example |
|--------|---------|
| CSV / TSV | `data.csv`, `data.tsv` |
| JSON | `data.json` |
| Excel | `data.xlsx` |
| Parquet | `data.parquet` |
| Database | `postgresql://user:pass@host/db` + `--db-table` |

### Options

```
opentaxonomy create
  -i, --input        Input file or database connection string  [required]
  -c, --column       Column containing raw values to classify  [required]
  -o, --output-dir   Where to write taxonomy files  [default: ./taxonomy]
  --domain-hint      Optional hint to guide the LLM (e.g. "German grocery products")
  --model            Claude model  [default: claude-sonnet-4-6]
  --api-key          Anthropic API key (or set ANTHROPIC_API_KEY)

opentaxonomy run
  -i, --input        Input file or database connection string  [required]
  -c, --column       Column containing raw values to classify  [required]
  -s, --seed-dir     Directory with seed.yaml and placement_map.yaml  [default: ./taxonomy]
  --model            Claude model  [default: claude-sonnet-4-6]
  --api-key          Anthropic API key (or set ANTHROPIC_API_KEY)
```

## Output structure

```
taxonomy/
├── seed.yaml                  # Domain seed: context, levels, edge cases
├── placement_map.yaml         # Raw values → canonical IDs (source of truth)
└── nodes/
    ├── root.yaml              # Root node
    ├── expenditure.yaml       # Internal node with decision record
    ├── groceries.yaml         # Leaf node with inclusion/exclusion criteria
    └── ...
```

Each node file is an ontological contract:

```yaml
node: Groceries
canonical_id: ft.expenditure.variable.groceries
question: Is this a purchase of food or household staples from a supermarket or grocery store?
criteria:
  includes:
    - Supermarket purchases (REWE, LIDL, EDEKA, etc.)
    - Organic/bio market purchases
  excludes:
    - Restaurant meals
    - Drugstore purchases unless food items
edge_cases:
  - term: REWE TO GO
    resolution: Included — still a grocery/convenience purchase
    decided: true
parent: Variable Necessities
children: []
version: 1.0.0
```

## License

MIT
