import hashlib
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


def content_hash(question: str, includes: list[str]) -> str:
    """SHA256[:12] of (question + sorted includes). Stable identity across renames."""
    text = question.strip().lower() + "|" + "|".join(sorted(s.lower() for s in includes))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def safe_child_id(label: str, parent_id: str, existing_ids: set[str]) -> str:
    """Generate a conflict-free canonical ID for a new child node."""
    base = slugify(label)
    candidate = f"{parent_id}.{base}"
    if candidate not in existing_ids:
        return candidate
    # Suffix with short hash of label to disambiguate
    suffix = hashlib.sha256(label.encode()).hexdigest()[:6]
    return f"{parent_id}.{base}_{suffix}"
