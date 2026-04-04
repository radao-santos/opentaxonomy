import re


def slugify(text: str) -> str:
    """'Variable Necessities' → 'variable_necessities'"""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s\-]+", "_", text)
    return text.strip("_")


def make_canonical_id(prefix: str, path: list[str]) -> str:
    """Build dotted canonical ID: make_canonical_id('ft', ['expenditure', 'variable']) → 'ft.expenditure.variable'"""
    parts = [prefix] + [slugify(p) for p in path]
    return ".".join(parts)
