SYSTEM_PRIMA_SEED = """\
You are executing the Prima Seed protocol — OpenTaxonomy's dialectical taxonomy generation engine.

The Prima Seed protocol (prima-seed.yaml):
- Step 0  Normalize: deduplicate, strip metadata, extract entity names
- Q0      Form recognition: the shared nature of all values → root node
- Q0b     Context establishment: the operational realm that governs all splits
- Q1      Primary differentiation: the most essential criterion dividing values into groups
- Q2      Recursive differentiation: applied per branch at each level
- Q3      Dialectical check: values that resist placement expose flawed criteria

Core principles:
- Context governs everything — the same data in different contexts produces different trees
- Criteria must be ontological (what a thing IS), not merely descriptive
- Edge cases are valuable: they mark where criteria need tightening
- Canonical IDs use dotted paths: prefix.level1.level2.leaf
- Every non-root node answers a yes/no question to determine placement
"""

SYSTEM_PLACER = """\
You are a semantic placement engine for OpenTaxonomy.
Given an existing taxonomy and a set of normalized values, assign each value to its correct leaf node.
Use the node criteria (includes/excludes) and decision records to guide placement.
When a value is genuinely ambiguous or unrecognisable, flag it as unresolved with a clear reason.
"""


def normalize_prompt(raw_values: list[str], domain_hint: str = "") -> str:
    values_str = "\n".join(f"  - {v}" for v in raw_values)
    hint = f"\nDomain hint: {domain_hint}" if domain_hint else ""
    return f"""\
Apply Step 0 of the Prima Seed protocol: normalize these raw values.{hint}

For each entity:
1. Strip metadata — dates, amounts, IBANs, reference numbers, transaction codes
2. Resolve encoding artifacts and spelling variants
3. Extract the core entity name (merchant, product, person, category, etc.)
4. Group raw values that refer to the same entity under one normalized name
5. Write a definition: 1-2 sentences describing WHAT this entity IS — not what category it belongs to,
   but what it fundamentally is. Examples:
   - "Butter Croissant" → "A flaky, layered pastry made with butter, typically eaten as a breakfast item or snack."
   - "Clean Wipes" → "Disposable pre-moistened cloths used for cleaning surfaces or personal hygiene."
   - "REWE" → "A major German supermarket chain selling food, beverages, and household goods."
   - "Netflix" → "A subscription-based video streaming platform offering films and TV series."

Raw values:
{values_str}
"""


def foundation_prompt(normalized_values: list[str], sample_size: int = 100) -> str:
    sample = normalized_values[:sample_size]
    values_str = "\n".join(f"  - {v}" for v in sample)
    note = f"\n(Showing {len(sample)} of {len(normalized_values)} unique values)" if len(normalized_values) > sample_size else ""
    return f"""\
Apply Q0 and Q0b of the Prima Seed protocol.{note}

Normalized values:
{values_str}

Determine:
1. root_label      — Q0: the highest-level Form unifying ALL these values
2. context_Q0      — Q0 answer in full (1–2 sentences)
3. context_Q0b     — Q0b: the operational context governing classification (2–3 sentences)
4. seed_id         — a short kebab-case identifier (e.g. 'personal-finance-transactions')
5. canonical_prefix — 2–4 letter abbreviation for canonical IDs (e.g. 'ft')
6. description     — one paragraph describing this domain seed
7. normalization_rules — list of normalization operations applied (e.g. 'extract_merchant_from_sepa')
"""


def primary_differentiation_prompt(
    normalized_values: list[str], root_label: str, context: str
) -> str:
    values_str = "\n".join(f"  - {v}" for v in normalized_values)
    return f"""\
Apply Q1 of the Prima Seed protocol.

Root: {root_label}
Context: {context}

Normalized values:
{values_str}

Q1: Within this context, what is the MOST ESSENTIAL criterion dividing these values into groups?

ABSTRACTION RULE — this is mandatory:
Every branch you produce must answer the SAME question at the SAME level of abstraction.
If one branch is broad (e.g. "Animal-Derived Products"), ALL sibling branches must be equally broad.
A specific sub-type (e.g. "Beer & Ales", "Frozen Desserts") cannot be a sibling of a broad category
(e.g. "Food & Drink") — that specific type belongs INSIDE the broad category, not alongside it.
If you find yourself creating one broad branch and several narrow ones, your criterion is wrong —
choose a broader criterion that produces branches of consistent scope.
Aim for 3–7 branches. Fewer broad branches are better than many specific ones.

For each branch provide:
- key              — short slug (no spaces, e.g. 'expenditure')
- label            — human-readable name (e.g. 'Expenditure')
- dimension        — the dimension being split on
- question         — the yes/no question that places something here
- members          — which normalized values belong to this branch
- criteria_includes — what belongs here
- criteria_excludes — what does NOT belong here
"""


def recursive_differentiation_prompt(
    branch_label: str, members: list[str], context: str, depth: int, max_depth: int = 4
) -> str:
    members_str = "\n".join(f"  - {v}" for v in members)
    levels_remaining = max_depth - depth
    return f"""\
Apply Q2 of the Prima Seed protocol to this branch (depth {depth} of {max_depth}).

Branch: {branch_label}
Context: {context}

Members:
{members_str}

Q2: What is the next most essential criterion within this branch?

ABSTRACTION RULE — this is mandatory:
All sub-branches produced here must answer the SAME question at the SAME level of abstraction.
A specific product type (e.g. "Cheddar Cheese") cannot be a sibling of a broad group
(e.g. "Dairy Products") — the specific type belongs INSIDE the broad group.
If your members are too similar to split meaningfully, return is_leaf=true rather than
creating artificially fine-grained branches.
Aim for 2–6 sub-branches of consistent scope.

DEPTH RULE — this is mandatory:
You are at depth {depth}. There {'is' if levels_remaining == 1 else 'are'} {levels_remaining} \
level{'s' if levels_remaining != 1 else ''} remaining before the maximum depth.
- If {levels_remaining} level{'s remain' if levels_remaining != 1 else ' remains'}: \
{'you MUST split unless ALL members are truly identical and indistinguishable by any criterion.' if levels_remaining > 0 else 'return is_leaf=true.'}
- Do NOT return is_leaf=true simply because there are few members — even 2 distinct values
  can be split into 2 meaningful sub-branches.
- A branch that returns is_leaf=true at depth {depth} will terminate earlier than its
  siblings, producing an uneven tree. Only do this if there is genuinely no criterion
  that separates the members in any meaningful way.

If values are sufficiently similar that no meaningful split exists, return is_leaf=true.

Otherwise, return sub-branches with:
- key, label, dimension, question
- members (which values go here)
- criteria_includes, criteria_excludes
"""


def dialectical_check_prompt(unplaced: list[str], tree_summary: str) -> str:
    unplaced_str = "\n".join(f"  - {v}" for v in unplaced)
    return f"""\
Apply Q3 of the Prima Seed protocol — the dialectical check.

Current tree:
{tree_summary}

Values resisting placement:
{unplaced_str}

For each resistant value:
1. Try harder — find its best-fit node in the existing tree
2. If it truly does not fit: record it as an edge case with the boundary it exposes
3. If several values reveal a structural flaw: recommend tree restructuring

Return: placed values, edge cases, and restructuring recommendations (if any).
"""


def node_details_prompt(
    label: str,
    parent_label: str,
    context: str,
    members: list[str],
    children_labels: list[str],
) -> str:
    members_str = "\n".join(f"  - {v}" for v in members) if members else "  (internal node — no direct values)"
    children_str = ", ".join(children_labels) if children_labels else "none (leaf)"
    return f"""\
Generate the full ontological definition for this taxonomy node.

Node: {label}
Parent: {parent_label}
Children: {children_str}
Context: {context}

Values assigned here:
{members_str}

Generate:
1. question         — the yes/no question answered 'yes' to place something here (specific, unambiguous)
2. criteria_includes — what belongs here (3–6 bullet points)
3. criteria_excludes — what does NOT belong here even if superficially similar (3–5 bullet points)
4. edge_cases       — values that test the boundary, with resolution and decided flag
5. decision_record  — why this criterion was chosen over alternatives (criterion_chosen, alternatives_considered, reason)
"""


def placement_prompt(normalized_values: list[str], tree_yaml: str) -> str:
    values_str = "\n".join(f"  - {v}" for v in normalized_values)
    return f"""\
Place these normalized values into the taxonomy.

Taxonomy:
{tree_yaml}

Values to place:
{values_str}

For each value:
- Assign it to the most precise leaf canonical_id
- List the raw_samples (use the normalized value itself if you have no other raw form)
- If genuinely unresolvable, flag it with a reason
"""


def tree_fit_prompt(values: list[str], node_label: str, node_canonical_id: str, context: str, children: list[dict]) -> str:
    values_str = "\n".join(f"  - {v}" for v in values)
    children_str = "\n".join(
        f"  - {c['canonical_id']} | {c['label']} | Q: {c['question']} | Includes: {', '.join(c['includes'][:2])}"
        for c in children
    )
    return f"""\
You are traversing a taxonomy tree to find where new values should be grafted.

Context: {context}
Current node: {node_label} ({node_canonical_id})

Existing child branches:
{children_str}

Values to classify:
{values_str}

For each value, decide:
- If it clearly fits under one of the existing children: return that child's canonical_id
- If it does not fit any existing child: return null (a new branch is needed here)

Be strict — only return a matching_canonical_id if the value genuinely belongs under that branch's criteria.
"""


def graft_prompt(values: list[str], parent_label: str, parent_question: str, parent_includes: list[str], context: str) -> str:
    values_str = "\n".join(f"  - {v}" for v in values)
    includes_str = "\n".join(f"  - {inc}" for inc in parent_includes)
    return f"""\
New values have arrived that do not fit any existing branch under "{parent_label}".

Context: {context}
Parent node: {parent_label}
Parent criterion: {parent_question}
Parent includes:
{includes_str}

Values to classify:
{values_str}

Create the minimum number of new branches needed to place all these values.
Group related values together — don't create one branch per value.

For each branch:
- key: short slug (no spaces)
- label: human-readable name
- question: yes/no placement question (specific, unambiguous)
- criteria_includes: what belongs here
- criteria_excludes: what does NOT belong here
- members: which of the above values go here
- edge_cases: any boundary cases worth noting

If a value truly cannot be classified even in a new branch, put it in unplaced.
"""
