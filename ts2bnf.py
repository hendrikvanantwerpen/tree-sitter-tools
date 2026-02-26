#!/usr/bin/env python3

import json
import sys


def load_grammar(path):
    with open(path) as f:
        return json.load(f)


def classify_rules(grammar):
    """Return (main_names: list, helper_set: set).

    Main rules: names NOT starting with '_', NOT in supertypes, NOT in inline.
    Helper rules: names starting with '_', OR in supertypes, OR in inline.
    """
    rules = grammar.get("rules", {})
    supertypes = set(grammar.get("supertypes", []))
    inline_list = set(grammar.get("inline", []))

    main_names = []
    helper_set = set()

    for name in rules:
        if name.startswith("_") or name in supertypes or name in inline_list:
            helper_set.add(name)
        else:
            main_names.append(name)

    return main_names, helper_set


def direct_refs_ordered(rule):
    """Return SYMBOL names in DFS order, deduplicated (first-occurrence).

    Follows SEQ/CHOICE/REPEAT/PREC/FIELD/etc. but does NOT follow ALIAS content.
    """
    seen = set()
    result = []

    def walk(r):
        t = r["type"]
        if t == "SYMBOL":
            name = r["name"]
            if name not in seen:
                seen.add(name)
                result.append(name)
        elif t in ("SEQ", "CHOICE"):
            for m in r["members"]:
                walk(m)
        elif t in ("REPEAT", "REPEAT1", "FIELD", "TOKEN", "IMMEDIATE_TOKEN",
                   "RESERVED", "PREC", "PREC_LEFT", "PREC_RIGHT", "PREC_DYNAMIC",
                   "ALIAS"):
            walk(r["content"])
        # BLANK, STRING, PATTERN — no symbols to recurse into

    walk(rule)
    return result


# Populated in main() before any formatting; used by format_inline to render
# helper symbol references as <name> instead of bare name.
_helper_set: set = set()

TRANSPARENT_TYPES = frozenset({
    "PREC", "PREC_LEFT", "PREC_RIGHT", "PREC_DYNAMIC",
    "TOKEN", "IMMEDIATE_TOKEN", "RESERVED",
})


def strip_transparent(rule):
    """Strip transparent wrapper types until a non-transparent type is found."""
    while rule["type"] in TRANSPARENT_TYPES:
        rule = rule["content"]
    return rule


def is_blank(rule):
    """Return True if rule is (or strips to) BLANK."""
    return strip_transparent(rule)["type"] == "BLANK"


def format_choice_expr(members):
    """Format a CHOICE's member list inline.

    Returns "(a | b)", "x?", "(a | b)?", or "" etc.
    """
    non_blank = [m for m in members if not is_blank(m)]
    has_blank = len(non_blank) < len(members)

    if not non_blank:
        return ""

    if len(non_blank) == 1:
        base = format_atom(non_blank[0]) if has_blank else format_inline(non_blank[0])
        return base + ("?" if has_blank else "")

    inner = " | ".join(format_inline(m) for m in non_blank)
    return "(" + inner + ")" + ("?" if has_blank else "")


def format_inline(rule):
    """Format a rule as an inline expression (no line breaks)."""
    t = rule["type"]

    if t == "BLANK":
        return ""
    elif t == "STRING":
        return '"' + rule["value"] + '"'
    elif t == "PATTERN":
        return "/" + rule["value"] + "/" + rule.get("flags", "")
    elif t == "SYMBOL":
        n = rule["name"]
        return "<" + n + ">" if n in _helper_set else n
    elif t == "ALIAS":
        content = format_inline(rule["content"])
        if rule.get("named", False):
            return content + "@" + rule["value"]
        else:
            return content + "@\"" + rule["value"] + "\""
    elif t == "FIELD":
        inner = format_inline(rule["content"])
        return rule["name"] + ":" + inner
    elif t == "SEQ":
        parts = [format_inline(m) for m in rule["members"]]
        return " ".join(p for p in parts if p)
    elif t == "CHOICE":
        return format_choice_expr(rule["members"])
    elif t == "REPEAT":
        return format_atom(rule["content"]) + "*"
    elif t == "REPEAT1":
        return format_atom(rule["content"]) + "+"
    elif t in TRANSPARENT_TYPES:
        return format_inline(rule["content"])
    else:
        return t  # fallback for unknown types


def format_atom(rule):
    """Format for use as REPEAT/REPEAT1 content; wrap compound nodes in parens.

    CHOICE already adds its own parens via format_choice_expr, so it is safe
    without an extra wrapper. SEQ and FIELD need explicit wrapping.
    """
    t = rule["type"]

    # These produce output that is safe as an atom without extra wrapping:
    #   - BLANK/STRING/PATTERN/SYMBOL/ALIAS: single tokens
    #   - REPEAT/REPEAT1: already end with * or +
    #   - CHOICE: format_choice_expr adds parens when needed
    if t in ("BLANK", "STRING", "PATTERN", "SYMBOL", "ALIAS",
             "REPEAT", "REPEAT1", "CHOICE"):
        return format_inline(rule)

    # Transparent wrappers: delegate to content
    if t in TRANSPARENT_TYPES:
        return format_atom(rule["content"])

    # SEQ and FIELD produce space-separated or colon-separated output
    # that needs parens to receive a * or + suffix as a unit.
    return "(" + format_inline(rule) + ")"


def compute_top_level_alts(rule):
    """Return list of alternative strings for top-level (multi-line) formatting."""
    stripped = strip_transparent(rule)

    if stripped["type"] != "CHOICE":
        s = format_inline(stripped)
        return [s] if s else ["(empty)"]

    seen = set()
    alts = []
    has_blank = False

    for m in stripped["members"]:
        if is_blank(m):
            has_blank = True
            continue
        s = format_inline(m)
        if s not in seen:
            seen.add(s)
            alts.append(s)

    if has_blank:
        alts.append("(empty)")

    return alts if alts else ["(empty)"]


def format_top_level(name, rule, is_helper=False):
    """Format a single rule as a multi-line BNF definition.

    Main rules use ':=', helper rules (_, inline, supertype) use '=='.
    Continuation '|' aligns with the second '=' in either separator.
    """
    alts = compute_top_level_alts(rule)
    sep = " == " if is_helper else " := "
    # '|' aligns with the second '=' (at column len(name)+2).
    # Example: "name := alt1"   or   "name == alt1"
    #          "      | alt2"        "      | alt2"
    padding = " " * (len(name) + 2)
    lines = [name + sep + alts[0]]
    for alt in alts[1:]:
        lines.append(padding + "| " + alt)
    return "\n".join(lines)


def print_rule_group(name, rule, rules, helper_set, printed_in_group):
    """Return list of output lines for a main rule and all its helper rules (DFS).

    Helpers are printed immediately after the rule that first references them.
    printed_in_group tracks what has already been emitted in this group to
    prevent duplicates from diamond references and to break cycles.

    All helpers are indented by the same amount regardless of nesting depth:
    their names start at the column of the '=' sign in the main rule.
    """
    output = []

    output.append(format_top_level(name, rule, is_helper=False))
    output.append("")  # blank line after every rule

    # Helpers align their names with the start of ':=' (the ':' column).
    # "name := ..." → ':' is at column len(name)+1, so indent = len(name)+1 spaces.
    indent = " " * (len(name) + 1)

    def emit_helpers(r):
        for ref in direct_refs_ordered(r):
            if ref not in helper_set:
                continue  # not a helper; main rule reference is fine as-is
            if ref not in rules:
                continue  # external symbol (e.g. _newline) — no rule body
            if ref in printed_in_group:
                continue  # already printed in this group (diamond or cycle)
            printed_in_group.add(ref)
            formatted = format_top_level(ref, rules[ref], is_helper=True)
            indented = "\n".join(indent + line for line in formatted.splitlines())
            output.append(indented)
            output.append("")
            emit_helpers(rules[ref])

    emit_helpers(rule)
    return output


def main():
    args = sys.argv[1:]
    expand = "--expand" in args
    paths = [a for a in args if not a.startswith("--")]

    if not paths:
        print("Convert a tree-sitter grammar.json to BNF format", file=sys.stderr)
        print("Usage: python3 ts2bnf.py [--expand] path/to/grammar.json", file=sys.stderr)
        sys.exit(1)

    grammar = load_grammar(paths[0])
    rules = grammar.get("rules", {})
    main_names, helper_set = classify_rules(grammar)

    global _helper_set
    _helper_set = helper_set

    output_lines = [
        f"# BNF grammar for {grammar.get("name")}",
        "#",
        "# Format:",
        "#",
        "#  - Node rules: defined with ':=', appear as nodes in the parse tree",
        "#  - Inlined rules: defined with '==' and wrapped in '<' and '>' at use site",
        "#  - Operators: '|' is choice, '*' is repetition, '+' is non-empty repetition",
        "#  - Fields: NAME:...",
        "#  - Aliases: ...@NAME",
    ]
    if expand:
        output_lines += [
        "#",
        "# The grammar is expanded, which means all inlined rules are included at their use",
        "# sites and omitted from the top level.",
        ]
    output_lines += [""]

    if expand:
        # --expand: each main rule is followed by its helper rules, indented and
        # co-located.  Helpers that are unreachable from any main rule are omitted.
        for name in main_names:
            rule = rules[name]
            printed_in_group = {name}
            group = print_rule_group(name, rule, rules, helper_set, printed_in_group)
            output_lines.extend(group)
    else:
        # Standard: every rule (main and helper) is printed at the top level in
        # grammar.json order, without indentation or grouping.
        for name, rule in rules.items():
            output_lines.append(format_top_level(name, rule, is_helper=name in helper_set))
            output_lines.append("")

    print("\n".join(output_lines).rstrip())


if __name__ == "__main__":
    main()
