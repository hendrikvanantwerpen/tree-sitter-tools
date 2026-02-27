#!/usr/bin/env python3

from dataclasses import dataclass, field
import json
import sys
from typing import Any, TypeAlias

TRANSPARENT_TYPES = frozenset({
    "PREC", "PREC_LEFT", "PREC_RIGHT", "PREC_DYNAMIC",
    "TOKEN", "IMMEDIATE_TOKEN", "RESERVED",
})


# ---------------------------------------------------------------------------
# Rule Data Types
# ---------------------------------------------------------------------------

@dataclass
class BlankRule:
    pass


@dataclass
class StringRule:
    value: str


@dataclass
class PatternRule:
    value: str
    flags: str = ""


@dataclass
class SymbolRule:
    name: str


@dataclass
class AliasRule:
    content: "RuleNode"
    value: str
    named: bool


@dataclass
class FieldRule:
    name: str
    content: "RuleNode"


@dataclass
class SeqRule:
    members: list["RuleNode"]


@dataclass
class ChoiceRule:
    members: list["RuleNode"]


@dataclass
class RepeatRule:
    content: "RuleNode"


@dataclass
class Repeat1Rule:
    content: "RuleNode"


@dataclass
class UnknownRule:
    kind: str


RuleNode: TypeAlias = (
    BlankRule
    | StringRule
    | PatternRule
    | SymbolRule
    | AliasRule
    | FieldRule
    | SeqRule
    | ChoiceRule
    | RepeatRule
    | Repeat1Rule
    | UnknownRule
)
RulesMap: TypeAlias = dict[str, RuleNode]


# ---------------------------------------------------------------------------
# Core Models / Runtime Context
# ---------------------------------------------------------------------------

@dataclass
class GrammarModel:
    name: str
    rules: RulesMap
    rule_order: list[str]
    inline_set: set[str]
    supertypes: set[str]
    inline_enabled: bool = False
    helper_refs_by_rule: dict[str, list[str]] = field(default_factory=dict)

    def is_inlinable(self, name: str) -> bool:
        return name.startswith("_") or name in self.supertypes or name in self.inline_set

    def main_rule_names(self) -> list[str]:
        return [name for name in self.rule_order if not self.is_inlinable(name)]


@dataclass
class RenderContext:
    model: GrammarModel
    inline_enabled: bool
    inline_candidate_set: set[str]
    inline_rules: RulesMap
    inline_stack: set[str] = field(default_factory=set)

    def is_inline_symbol(self, name: str) -> bool:
        return (
            self.inline_enabled
            and name in self.inline_candidate_set
            and name in self.inline_rules
            and name not in self.inline_stack
        )

    def is_helper_symbol(self, name: str) -> bool:
        return self.model.is_inlinable(name)


# ---------------------------------------------------------------------------
# Transformation 0: Parse JSON -> Typed GrammarModel
# ---------------------------------------------------------------------------

def load_grammar(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def parse_rule(raw: dict[str, Any]) -> RuleNode:
    rule_type = raw.get("type", "UNKNOWN")
    if rule_type in TRANSPARENT_TYPES:
        return parse_rule(raw["content"])

    match rule_type:
        case "BLANK":
            return BlankRule()
        case "STRING":
            return StringRule(value=raw["value"])
        case "PATTERN":
            return PatternRule(value=raw["value"], flags=raw.get("flags", ""))
        case "SYMBOL":
            return SymbolRule(name=raw["name"])
        case "ALIAS":
            return AliasRule(
                content=parse_rule(raw["content"]),
                value=raw["value"],
                named=bool(raw.get("named", False)),
            )
        case "FIELD":
            return FieldRule(name=raw["name"], content=parse_rule(raw["content"]))
        case "SEQ":
            return SeqRule(members=[parse_rule(member) for member in raw.get("members", [])])
        case "CHOICE":
            return ChoiceRule(members=[parse_rule(member) for member in raw.get("members", [])])
        case "REPEAT":
            return RepeatRule(content=parse_rule(raw["content"]))
        case "REPEAT1":
            return Repeat1Rule(content=parse_rule(raw["content"]))
        case _:
            return UnknownRule(kind=str(rule_type))


def parse_grammar(path: str) -> GrammarModel:
    """Transformation 0: parse JSON into a grammar model."""
    grammar = load_grammar(path)
    raw_rules = grammar.get("rules", {})
    return GrammarModel(
        name=grammar.get("name", ""),
        rules={name: parse_rule(rule) for name, rule in raw_rules.items()},
        rule_order=list(raw_rules.keys()),
        inline_set=set(grammar.get("inline", [])),
        supertypes=set(grammar.get("supertypes", [])),
    )


# ---------------------------------------------------------------------------
# Shared Rule Utilities
# ---------------------------------------------------------------------------

def is_blank(rule: RuleNode) -> bool:
    return isinstance(rule, BlankRule)


def direct_refs_ordered(rule: RuleNode) -> list[str]:
    """Return SYMBOL names in DFS order, deduplicated (first-occurrence)."""
    seen: set[str] = set()
    result: list[str] = []

    def walk(node: RuleNode) -> None:
        match node:
            case SymbolRule(name=name):
                if name not in seen:
                    seen.add(name)
                    result.append(name)
            case SeqRule(members=members) | ChoiceRule(members=members):
                for member in members:
                    walk(member)
            case AliasRule(content=content) | FieldRule(content=content) | RepeatRule(content=content) | Repeat1Rule(content=content):
                walk(content)
            case _:
                return

    walk(rule)
    return result


def compute_inline_candidates(model: GrammarModel) -> set[str]:
    """Return helper names that are safe to inline."""
    candidates: set[str] = set()
    for name in model.rule_order:
        if name not in model.rules or not model.is_inlinable(name):
            continue
        if not isinstance(model.rules[name], ChoiceRule):
            candidates.add(name)
    return candidates


# ---------------------------------------------------------------------------
# Transformation 1: Collect Helper Rule References
# ---------------------------------------------------------------------------

def collect_helper_refs(model: GrammarModel) -> GrammarModel:
    """Transformation 1: collect helper refs for each rule."""
    refs: dict[str, list[str]] = {}
    for name, rule in model.rules.items():
        refs[name] = [ref for ref in direct_refs_ordered(rule) if model.is_inlinable(ref)]
    model.helper_refs_by_rule = refs
    return model


# ---------------------------------------------------------------------------
# Transformation 2: Configure Optional Inlining
# ---------------------------------------------------------------------------

def configure_inlining(model: GrammarModel, inline_enabled: bool) -> set[str]:
    """Transformation 2: configure inlining and remove inlined helpers per rule."""
    model.inline_enabled = inline_enabled
    inline_candidates = compute_inline_candidates(model) if inline_enabled else set()

    refs: dict[str, list[str]] = {}
    for name, rule_refs in model.helper_refs_by_rule.items():
        refs[name] = [
            ref for ref in rule_refs
            if not (inline_enabled and ref in inline_candidates and ref in model.rules)
        ]
    model.helper_refs_by_rule = refs
    return inline_candidates


# ---------------------------------------------------------------------------
# Transformation 3: Lift First-Level Complex Choices
# ---------------------------------------------------------------------------

def _complexity_target(rule: RuleNode, rules: RulesMap, seen: set[str]) -> RuleNode | None:
    match rule:
        case SymbolRule(name=name):
            if name in seen or name not in rules:
                return None
            seen.add(name)
            return _complexity_target(rules[name], rules, seen)
        case AliasRule(content=content) | FieldRule(content=content) | RepeatRule(content=content) | Repeat1Rule(content=content):
            return _complexity_target(content, rules, seen)
        case _:
            return rule


def is_complex_choice_member(rule: RuleNode, rules: RulesMap) -> bool:
    """Return True if member resolves to a multi-item SEQ or CHOICE."""
    target = _complexity_target(rule, rules, set())
    match target:
        case SeqRule(members=members) | ChoiceRule(members=members):
            non_blank = [member for member in members if not is_blank(member)]
            return len(non_blank) > 1
        case _:
            return False


def _choice_from_seq_member(rule: RuleNode) -> tuple[int | None, ChoiceRule | None]:
    """Return (choice_index, choice_rule) for a first-level CHOICE in top SEQ."""
    if not isinstance(rule, SeqRule):
        return None, None
    for idx, member in enumerate(rule.members):
        if isinstance(member, ChoiceRule):
            return idx, member
    return None, None


def _lift_choice_in_seq(seq_rule: RuleNode, rules: RulesMap) -> RuleNode | None:
    idx, choice = _choice_from_seq_member(seq_rule)
    if choice is None or idx is None:
        return None

    if not any(is_complex_choice_member(member, rules) for member in choice.members):
        return None

    seq_members = seq_rule.members if isinstance(seq_rule, SeqRule) else []
    alts: list[RuleNode] = []
    for member in choice.members:
        new_members = list(seq_members)
        if is_blank(member):
            del new_members[idx]
        else:
            new_members[idx] = member

        if len(new_members) == 0:
            alt: RuleNode = BlankRule()
        elif len(new_members) == 1:
            alt = new_members[0]
        else:
            alt = SeqRule(members=new_members)
        alts.append(alt)

    return ChoiceRule(members=alts)


def _rebuild_with_content(wrapper: RuleNode, content: RuleNode) -> RuleNode | None:
    match wrapper:
        case AliasRule(value=value, named=named):
            return AliasRule(content=content, value=value, named=named)
        case FieldRule(name=name):
            return FieldRule(name=name, content=content)
        case RepeatRule():
            return RepeatRule(content=content)
        case Repeat1Rule():
            return Repeat1Rule(content=content)
        case _:
            return None


def _lift_choice_in_wrapped_content(rule: RuleNode, rules: RulesMap) -> RuleNode | None:
    match rule:
        case AliasRule(content=content) | FieldRule(content=content) | RepeatRule(content=content) | Repeat1Rule(content=content):
            pass
        case _:
            return None

    lifted: RuleNode | None = None
    match content:
        case ChoiceRule(members=members):
            if any(is_complex_choice_member(member, rules) for member in members):
                lifted = content
        case SeqRule():
            lifted = _lift_choice_in_seq(content, rules)
        case _:
            lifted = None

    if not isinstance(lifted, ChoiceRule):
        return None

    wrapped: list[RuleNode] = []
    for member in lifted.members:
        rebuilt = _rebuild_with_content(rule, member)
        if rebuilt is None:
            return None
        wrapped.append(rebuilt)
    return ChoiceRule(members=wrapped)


def lift_first_level_choice(rule: RuleNode, rules: RulesMap) -> RuleNode:
    """Transformation 3: lift first-level complex CHOICE to top-level CHOICE."""
    if isinstance(rule, SeqRule):
        lifted = _lift_choice_in_seq(rule, rules)
        return lifted if lifted else rule

    lifted_wrapped = _lift_choice_in_wrapped_content(rule, rules)
    if lifted_wrapped:
        return lifted_wrapped

    return rule


def lift_choices(model: GrammarModel) -> GrammarModel:
    """Transformation 3 over grammar model."""
    lifted_rules: RulesMap = {}
    for name, rule in model.rules.items():
        lifted_rules[name] = lift_first_level_choice(rule, model.rules)
    model.rules = lifted_rules
    return model


# ---------------------------------------------------------------------------
# Transformation 4: Render / Print Output
# ---------------------------------------------------------------------------

def is_inlinable_symbol(ctx: RenderContext, name: str) -> bool:
    return ctx.is_inline_symbol(name)


def format_inline_symbol(ctx: RenderContext, name: str) -> str:
    """Format an inlinable helper reference as name/body."""
    ctx.inline_stack.add(name)
    try:
        body_str = format_atom(ctx, ctx.inline_rules[name])
        return name + "/" + body_str
    finally:
        ctx.inline_stack.discard(name)


def format_choice_expr(ctx: RenderContext, members: list[RuleNode]) -> str:
    """Format a CHOICE's member list inline."""
    non_blank = [member for member in members if not is_blank(member)]
    has_blank = len(non_blank) < len(members)

    if not non_blank:
        return ""

    if len(non_blank) == 1:
        base = format_atom(ctx, non_blank[0]) if has_blank else format_inline(ctx, non_blank[0])
        return base + ("?" if has_blank else "")

    inner = " | ".join(format_inline(ctx, member) for member in non_blank)
    return "(" + inner + ")" + ("?" if has_blank else "")


def format_inline(ctx: RenderContext, rule: RuleNode) -> str:
    """Format a rule as an inline expression (no line breaks)."""
    match rule:
        case BlankRule():
            return ""
        case StringRule(value=value):
            return '"' + value + '"'
        case PatternRule(value=value, flags=flags):
            return "/" + value + "/" + flags
        case SymbolRule(name=name):
            if is_inlinable_symbol(ctx, name):
                return format_inline_symbol(ctx, name)
            if name in ctx.inline_stack:
                return name
            return "<" + name + ">" if ctx.is_helper_symbol(name) else name
        case AliasRule(content=content, value=value, named=named):
            rendered = format_inline(ctx, content)
            if named:
                return rendered + "@" + value
            return rendered + '@"' + value + '"'
        case FieldRule(name=name, content=content):
            return name + ":" + format_inline(ctx, content)
        case SeqRule(members=members):
            parts = [format_inline(ctx, member) for member in members]
            return " ".join(part for part in parts if part)
        case ChoiceRule(members=members):
            return format_choice_expr(ctx, members)
        case RepeatRule(content=content):
            return format_atom(ctx, content) + "*"
        case Repeat1Rule(content=content):
            return format_atom(ctx, content) + "+"
        case UnknownRule(kind=kind):
            return kind


def format_atom(ctx: RenderContext, rule: RuleNode) -> str:
    """Format for use as REPEAT/REPEAT1 content."""
    match rule:
        case SymbolRule(name=name):
            if is_inlinable_symbol(ctx, name):
                return "(" + format_inline_symbol(ctx, name) + ")"
            return format_inline(ctx, rule)
        case BlankRule() | StringRule() | PatternRule() | AliasRule() | RepeatRule() | Repeat1Rule() | ChoiceRule():
            return format_inline(ctx, rule)
        case _:
            return "(" + format_inline(ctx, rule) + ")"


def compute_top_level_alts(ctx: RenderContext, rule: RuleNode) -> list[str]:
    """Return list of alternative strings for top-level formatting."""
    if not isinstance(rule, ChoiceRule):
        rendered = format_inline(ctx, rule)
        return [rendered] if rendered else ["(empty)"]

    seen: set[str] = set()
    alts: list[str] = []
    has_blank = False
    for member in rule.members:
        if is_blank(member):
            has_blank = True
            continue
        rendered = format_inline(ctx, member)
        if rendered not in seen:
            seen.add(rendered)
            alts.append(rendered)

    if has_blank:
        alts.append("(empty)")
    return alts if alts else ["(empty)"]


def format_top_level(
    ctx: RenderContext,
    name: str,
    rule: RuleNode,
    is_helper: bool = False,
) -> str:
    """Format a single rule as a multi-line BNF definition."""
    alts = compute_top_level_alts(ctx, rule)
    sep = " == " if is_helper else " := "
    padding = " " * (len(name) + 2)
    lines = [name + sep + alts[0]]
    for alt in alts[1:]:
        lines.append(padding + "| " + alt)
    return "\n".join(lines)


def print_rule_group(
    ctx: RenderContext,
    model: GrammarModel,
    name: str,
    printed_in_group: set[str],
) -> list[str]:
    """Return lines for a main rule and its helper rules (DFS)."""
    output: list[str] = []
    output.append(format_top_level(ctx, name, model.rules[name], is_helper=False))
    output.append("")

    indent = " " * (len(name) + 1)

    def emit_helpers(rule_name: str) -> None:
        for ref in model.helper_refs_by_rule.get(rule_name, []):
            if ref not in model.rules:
                continue
            if ref in printed_in_group:
                continue
            printed_in_group.add(ref)
            formatted = format_top_level(ctx, ref, model.rules[ref], is_helper=True)
            output.append("\n".join(indent + line for line in formatted.splitlines()))
            output.append("")
            emit_helpers(ref)

    emit_helpers(name)
    return output


def build_header(model: GrammarModel, expand: bool) -> list[str]:
    output_lines: list[str] = [
        f"# BNF grammar for {model.name}",
        "#",
        "# Format:",
        "#",
        "#  - Node rules: defined with ':=', appear as nodes in the parse tree",
        "#  - Inlined rules: defined with '==' and wrapped in '<' and '>' at use site",
        "#  - Operators: '|' is choice, '*' is repetition, '+' is non-empty repetition",
        "#  - Fields: NAME:...",
        "#  - Aliases: ...@SYMBOL",
    ]
    if expand:
        output_lines += [
            "#",
            "# The grammar is expanded: helper rules are shown indented at their use sites",
            "# and omitted from the top level.",
        ]
    if model.inline_enabled:
        output_lines += [
            "#",
            "# The grammar is inlined: single alternative helper rules appear as symbol/body",
            "# instead of <symbol>.",
        ]
    output_lines.append("")
    return output_lines


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    expand = "--expand" in args
    inline = "--inline" in args
    paths = [arg for arg in args if not arg.startswith("--")]

    if not paths:
        print("Convert a tree-sitter grammar.json to BNF format", file=sys.stderr)
        print("Usage: python3 ts2bnf.py [--expand] [--inline] path/to/grammar.json", file=sys.stderr)
        sys.exit(1)

    model = parse_grammar(paths[0])
    collect_helper_refs(model)
    inline_candidate_set = configure_inlining(model, inline)
    inline_rules = dict(model.rules)
    lift_choices(model)

    ctx = RenderContext(
        model=model,
        inline_enabled=model.inline_enabled,
        inline_candidate_set=inline_candidate_set,
        inline_rules=inline_rules,
    )

    output_lines = build_header(model, expand)
    if expand:
        for name in model.main_rule_names():
            printed_in_group = {name}
            output_lines.extend(print_rule_group(ctx, model, name, printed_in_group))
    else:
        for name in model.rule_order:
            rule = model.rules[name]
            output_lines.append(format_top_level(ctx, name, rule, is_helper=model.is_inlinable(name)))
            output_lines.append("")

    print("\n".join(output_lines).rstrip())


if __name__ == "__main__":
    main()
