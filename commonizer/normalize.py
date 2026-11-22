"""Connector-name parsing, wire fingerprints and Vercodung (variant code rule) parsing."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import FrozenSet, List, Tuple

import pandas as pd

# A101aX13.3 -> letter=A num=101 var=a plug=13 pos=3 ; X308a.2 -> letter=X num=308 var=a pos=2
CONN_RE = re.compile(r"^(?P<letter>[A-Za-z]+?)(?P<num>\d+)(?P<var>[a-z]*)(?:X(?P<plug>\d+))?(?:\.(?P<pos>\d+))?$")


@lru_cache(maxsize=None)
def parse_connector(name: str) -> Tuple[str, str, str]:
    """Returns (class_letter, base, position). Unparseable names get letter '#'."""
    m = CONN_RE.match(name or "")
    if not m:
        pos = name.rsplit(".", 1)[1] if "." in (name or "") and name.rsplit(".", 1)[1].isdigit() else ""
        base = name.rsplit(".", 1)[0] if pos else name
        return "#" + (re.match(r"[A-Za-z]*", name or "").group() or ""), base, pos
    pos = m.group("pos") or ""
    base = name[: -(len(pos) + 1)] if pos else name
    return m.group("letter"), base, pos


def connector_block(name: str) -> str:
    """Blocking key for alignment: renames keep class letter and position suffix."""
    letter, _, pos = parse_connector(name)
    return f"{letter}|{pos}"


def wire_fingerprint(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Stable-attribute fingerprint, hashed to a short token."""
    cols = [c for c in cols if c in df.columns]
    joined = df[cols].astype(str).agg("\x1f".join, axis=1) if cols else pd.Series("", index=df.index)
    return joined.map(lambda s: hashlib.blake2b(s.encode(), digest_size=6).hexdigest())


def add_endpoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Order-insensitive endpoints: end1 <= end2 lexicographically."""
    a = df["connector_a"] + "/" + df["pin_a"]
    b = df["connector_b"] + "/" + df["pin_b"]
    swap = a > b
    df = df.copy()
    df["e1_conn"] = df["connector_a"].where(~swap, df["connector_b"])
    df["e1_pin"] = df["pin_a"].where(~swap, df["pin_b"])
    df["e2_conn"] = df["connector_b"].where(~swap, df["connector_a"])
    df["e2_pin"] = df["pin_b"].where(~swap, df["pin_a"])
    return df


# ---------------------------------------------------------------- Vercodung
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class CodeRule:
    """AND of terms; each term is (negated, frozenset(OR-codes))."""
    terms: Tuple[Tuple[bool, FrozenSet[str]], ...]

    @property
    def positive(self) -> FrozenSet[str]:
        return frozenset(c for neg, cs in self.terms if not neg for c in cs)

    @property
    def negative(self) -> FrozenSet[str]:
        return frozenset(c for neg, cs in self.terms if neg for c in cs)

    @property
    def required(self) -> FrozenSet[str]:
        """Codes that must be present (single-code positive terms)."""
        return frozenset(next(iter(cs)) for neg, cs in self.terms if not neg and len(cs) == 1)

    def canonical(self) -> str:
        parts = []
        for neg, cs in sorted(self.terms, key=lambda t: (t[0], sorted(t[1]))):
            body = "/".join(sorted(cs))
            body = f"({body})" if len(cs) > 1 else body
            parts.append(("-" if neg else "") + body)
        return "+".join(parts)

    def tokens(self) -> List[str]:
        out = []
        for neg, cs in self.terms:
            for c in cs:
                out.append(("N_" if neg else "P_") + c)
        return out


def _split_top(s: str, sep: str) -> List[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [p for p in (x.strip() for x in out) if p]


@lru_cache(maxsize=200_000)
def parse_vercodung(expr: str) -> CodeRule:
    """Parses rules like  -(D6V/B0L)+-L0L+(J9D/O3A)+I0Q ; '+' = AND, '/' = OR, '-' = NOT.

    Nested/unknown syntax degrades to a flat token set, never raises."""
    expr = (expr or "").replace(" ", "")
    terms = []
    for part in _split_top(expr, "+"):
        neg = part.startswith("-")
        body = part[1:] if neg else part
        body = body.strip("()")
        codes = frozenset(TOKEN_RE.findall(body))
        if codes:
            terms.append((neg, codes))
    return CodeRule(tuple(terms))


def rules_conflict(a: CodeRule, b: CodeRule) -> FrozenSet[str]:
    """Codes one rule requires that the other forbids (mutually exclusive variants)."""
    return (a.required & b.negative) | (b.required & a.negative)


def combine_rules(rules: List[CodeRule]) -> str:
    """Suggested combined rule for a merged part: common AND-terms, then OR of the remainders."""
    rules = [r for r in rules if r.terms]
    if not rules:
        return ""
    common = set(rules[0].terms)
    for r in rules[1:]:
        common &= set(r.terms)
    rest = []
    for r in rules:
        rem = CodeRule(tuple(t for t in r.terms if t not in common)).canonical()
        if rem and rem not in rest:
            rest.append(rem)
    head = CodeRule(tuple(common)).canonical()
    if not rest or len(rest) < len(rules):  # some rule is fully covered by the common part
        return head or "(always)"
    alt = " | ".join(f"[{x}]" for x in rest)
    return f"{head} + ({alt})" if head else alt
