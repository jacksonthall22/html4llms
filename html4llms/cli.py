#!/usr/bin/env python3
"""
Installable CLI that condenses DevTools "Copy > outerHTML" dumps into LLM-readable HTML:
- Condense <svg>...</svg> into a short placeholder
- Replace <script> and <style> contents with "..."
- Truncate long attribute values (e.g., huge class lists, base64 data URLs, long querystrings)
- Pretty-print with indentation
- Optional: limit depth / children per node

Usage:
  # clipboard in/out (default; pbpaste -> pbcopy)
  html4llms

  # from stdin
  pbpaste | html4llms > reduced.html
  cat big.html | html4llms --max-depth 18 --max-children 40 > reduced.html

  # from file
  html4llms --in big.html --out reduced.html

To run straight from a checkout without installing:
  uv run html4llms/cli.py --in big.html --out reduced.html
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit, parse_qsl


# --- Node model ----------------------------------------------------------------

REMOVAL_COMMENT = "<!-- (content removed) -->"
REMOVAL_HEADER_COMMENT = "<!-- For LLMs: some content below may have been removed for brevity. -->"


@dataclass
class TextNode:
    text: str


@dataclass
class ElementNode:
    tag: str
    attrs: Dict[str, Optional[str]] = field(default_factory=dict)
    children: List["Node"] = field(default_factory=list)


Node = Union[TextNode, ElementNode]


# --- Parsing -------------------------------------------------------------------

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}

RAW_TEXT_ELEMENTS = {"script", "style"}  # we want to keep tag but drop contents


class OuterHTMLParser(HTMLParser):
    """
    Build a simple DOM from HTML. Robust enough for DevTools outerHTML.
    """
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = ElementNode(tag="__root__")
        self.stack: List[ElementNode] = [self.root]
        self._raw_text_stack: List[str] = []  # tracks script/style nesting

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        node = ElementNode(tag=tag.lower(), attrs={k.lower(): v for k, v in attrs})
        self.stack[-1].children.append(node)

        if tag.lower() in RAW_TEXT_ELEMENTS:
            self._raw_text_stack.append(tag.lower())

        # HTMLParser doesn't auto-close voids; don't push them onto stack.
        if tag.lower() not in VOID_ELEMENTS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        node = ElementNode(tag=tag.lower(), attrs={k.lower(): v for k, v in attrs})
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()

        if self._raw_text_stack and self._raw_text_stack[-1] == t:
            self._raw_text_stack.pop()

        # Pop until matching tag found (best-effort)
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == t:
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        # If we're inside <script> or <style>, ignore contents entirely.
        if self._raw_text_stack:
            return
        if data:
            self.stack[-1].children.append(TextNode(data))

    def handle_comment(self, data: str) -> None:
        # Drop comments by default
        return


def parse_outer_html(s: str) -> ElementNode:
    p = OuterHTMLParser()
    p.feed(s)
    p.close()
    return p.root


# --- Simplification rules ------------------------------------------------------

DEFAULT_KEEP_ATTRS = {
    "id",
    "class",
    "role",
    "tabindex",
    "dir",
    "type",
    "name",
    "value",
    "href",
    "src",
    "alt",
    "title",
    "disabled",
    # Commonly useful on ChatGPT DOM
    "data-testid",
    "data-turn",
    "data-turn-id",
    "data-rc-turn-id",
    "data-state",
    "aria-label",
    "aria-pressed",
    "aria-expanded",
    "aria-haspopup",
}

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = WHITESPACE_RE.sub(" ", s.strip())
    return s


def shorten_middle(s: str, max_len: int) -> Tuple[str, bool]:
    if len(s) <= max_len:
        return s, False
    if max_len < 20:
        return s[: max_len - 3] + "...", True
    head = (max_len // 2) - 3
    tail = max_len - head - 3
    return f"{s[:head]}...{s[-tail:]}", True


def shorten_data_url(s: str, max_len: int) -> Tuple[str, bool]:
    # data:[<mediatype>][;base64],<data>
    if not s.startswith("data:"):
        return shorten_middle(s, max_len)
    prefix = s[: min(len(s), 80)]
    shortened, truncated_middle = shorten_middle(prefix, 80)
    return f"{shortened}...(data-url,len={len(s)})", truncated_middle or len(s) > len(prefix)


def shorten_class_value(class_val: str, max_classes: int) -> Tuple[str, bool]:
    parts = [p for p in class_val.split() if p]
    if len(parts) <= max_classes:
        return " ".join(parts), False
    kept = parts[:max_classes]
    return " ".join(kept) + f" ...( +{len(parts) - max_classes} )", True


def shorten_url(u: str, max_len: int, keep_query_params: int = 2) -> Tuple[str, bool]:
    if len(u) <= max_len:
        return u, False
    try:
        sp = urlsplit(u)
        base = f"{sp.scheme}://{sp.netloc}{sp.path}"
        if not sp.query:
            return shorten_middle(base, max_len)
        q = parse_qsl(sp.query, keep_blank_values=True)
        kept = q[:keep_query_params]
        q_str = "&".join(f"{k}={v}" for k, v in kept)
        suffix = f"?{q_str}"
        if len(q) > keep_query_params:
            suffix += f"...(+{len(q) - keep_query_params})"
        candidate = base + suffix
        return shorten_middle(candidate, max_len)
    except Exception:
        return shorten_middle(u, max_len)


def annotate_removal(s: str, removed: bool) -> str:
    if not removed:
        return s
    if REMOVAL_COMMENT in s:
        return s
    return f"{s} {REMOVAL_COMMENT}"


def simplify_attr_value(
    name: str,
    val: Optional[str],
    max_attr_len: int,
    max_classes: int,
) -> Tuple[Optional[str], bool]:
    if val is None:
        return None, False  # boolean attr

    v = normalize_text(val)

    if not v:
        return "", False

    # Special handling
    if name == "class":
        return shorten_class_value(v, max_classes=max_classes)

    if name in {"href", "src"}:
        if v.startswith("data:"):
            return shorten_data_url(v, max_attr_len)
        return shorten_url(v, max_len=max_attr_len)

    if v.startswith("data:"):
        return shorten_data_url(v, max_attr_len)

    return shorten_middle(v, max_attr_len)


def should_keep_attr(name: str, keep_style: bool, keep_data_attrs: bool) -> bool:
    if name == "style":
        return keep_style
    if name in DEFAULT_KEEP_ATTRS:
        return True
    if name.startswith("aria-"):
        return True
    if keep_data_attrs and name.startswith("data-"):
        return True
    return False


def format_attrs(
    attrs: Dict[str, Optional[str]],
    *,
    max_attr_len: int,
    max_classes: int,
    keep_style: bool,
    keep_data_attrs: bool,
) -> Tuple[str, bool]:
    kept_items: List[Tuple[str, Optional[str]]] = []
    for k, v in attrs.items():
        k = k.lower()
        if should_keep_attr(k, keep_style=keep_style, keep_data_attrs=keep_data_attrs):
            kept_items.append((k, v))

    # Stable order: common attrs first, then rest
    def sort_key(item: Tuple[str, Optional[str]]) -> Tuple[int, str]:
        k, _ = item
        priority = {
            "id": 0,
            "class": 1,
            "data-testid": 2,
            "data-turn-id": 3,
            "data-turn": 4,
            "role": 5,
            "aria-label": 6,
            "href": 7,
            "src": 8,
        }
        return (priority.get(k, 50), k)

    kept_items.sort(key=sort_key)

    parts: List[str] = []
    any_truncated = False
    for k, v in kept_items:
        sv, truncated_val = simplify_attr_value(k, v, max_attr_len=max_attr_len, max_classes=max_classes)
        any_truncated = any_truncated or truncated_val
        if sv is None:
            parts.append(k)  # boolean
        else:
            escaped = html.escape(sv, quote=True)
            parts.append(f'{k}="{escaped}"')
    return ((" " + " ".join(parts)) if parts else "", any_truncated)


def svg_summary(node: ElementNode) -> str:
    # Count descendants quickly
    stack: List[Node] = list(node.children)
    tags = 0
    paths = 0
    while stack:
        cur = stack.pop()
        if isinstance(cur, ElementNode):
            tags += 1
            if cur.tag == "path":
                paths += 1
            stack.extend(cur.children)
    return f"svg:children_tags={tags},paths={paths}"


# --- Rendering -----------------------------------------------------------------

def render(
    node: Node,
    *,
    depth: int,
    indent: str,
    max_depth: int,
    max_children: int,
    max_attr_len: int,
    max_text_len: int,
    max_classes: int,
    keep_style: bool,
    keep_data_attrs: bool,
) -> Tuple[List[str], bool]:
    pad = indent * depth

    if isinstance(node, TextNode):
        t = normalize_text(node.text)
        if not t:
            return [], False
        t, truncated_text = shorten_middle(t, max_text_len)
        line = pad + html.escape(t)
        return [annotate_removal(line, truncated_text)], truncated_text

    # Element
    tag = node.tag
    attrs_str, attrs_truncated = format_attrs(
        node.attrs,
        max_attr_len=max_attr_len,
        max_classes=max_classes,
        keep_style=keep_style,
        keep_data_attrs=keep_data_attrs,
    )
    removal_seen = attrs_truncated

    # Depth limit
    if depth >= max_depth:
        line = pad + f"<{tag}{attrs_str}>...(depth limit)</{tag}>"
        return [annotate_removal(line, True)], True

    # Special: script/style contents removed
    if tag in RAW_TEXT_ELEMENTS:
        line = pad + f"<{tag}{attrs_str}>...</{tag}>"
        return [annotate_removal(line, True)], True

    # Special: SVG condensed
    if tag == "svg":
        summ = svg_summary(node)
        line = pad + f"<svg{attrs_str}>...({summ})</svg>"
        return [annotate_removal(line, True)], True

    # Render children with limits
    meaningful_children = [c for c in node.children if not (isinstance(c, TextNode) and not normalize_text(c.text))]
    n = len(meaningful_children)

    def render_children(children: List[Node]) -> Tuple[List[str], bool]:
        out: List[str] = []
        removed_any = False
        for c in children:
            rendered, child_removed = render(
                c,
                depth=depth + 1,
                indent=indent,
                max_depth=max_depth,
                max_children=max_children,
                max_attr_len=max_attr_len,
                max_text_len=max_text_len,
                max_classes=max_classes,
                keep_style=keep_style,
                keep_data_attrs=keep_data_attrs,
            )
            out.extend(rendered)
            removed_any = removed_any or child_removed
        return out, removed_any

    if n == 0:
        if tag in VOID_ELEMENTS:
            line = pad + f"<{tag}{attrs_str} />"
            return [annotate_removal(line, attrs_truncated)], removal_seen
        line = pad + f"<{tag}{attrs_str}></{tag}>"
        return [annotate_removal(line, attrs_truncated)], removal_seen

    # Child limit
    if n > max_children:
        head_n = max_children // 2
        tail_n = max_children - head_n
        head = meaningful_children[:head_n]
        tail = meaningful_children[-tail_n:]
        head_lines, head_removed = render_children(head)
        tail_lines, tail_removed = render_children(tail)
        child_lines = head_lines
        child_lines.append(annotate_removal(pad + indent + f"...(+{n - (head_n + tail_n)} more children)", True))
        child_lines.extend(tail_lines)
        removed_children = True
    else:
        child_lines, removed_children = render_children(meaningful_children)

    removal_seen = removal_seen or removed_children

    # If it's a simple single text line, inline it
    if len(child_lines) == 1 and not child_lines[0].lstrip().startswith("<"):
        inline = child_lines[0].strip()
        line = pad + f"<{tag}{attrs_str}>{inline}</{tag}>"
        return [annotate_removal(line, attrs_truncated)], removal_seen

    lines: List[str] = [annotate_removal(pad + f"<{tag}{attrs_str}>", attrs_truncated)]
    lines.extend(child_lines)
    lines.append(pad + f"</{tag}>")
    return lines, removal_seen


def simplify_html(
    html_input: str,
    *,
    max_depth: int,
    max_children: int,
    max_attr_len: int,
    max_text_len: int,
    max_classes: int,
    keep_style: bool,
    keep_data_attrs: bool,
    indent: str,
) -> str:
    root = parse_outer_html(html_input)

    # If user pasted a single outerHTML node, root will contain that node as child.
    # Render all children of __root__ (usually just one).
    lines: List[str] = []
    removed_any = False
    for child in root.children:
        rendered, removed = render(
            child,
            depth=0,
            indent=indent,
            max_depth=max_depth,
            max_children=max_children,
            max_attr_len=max_attr_len,
            max_text_len=max_text_len,
            max_classes=max_classes,
            keep_style=keep_style,
            keep_data_attrs=keep_data_attrs,
        )
        lines.extend(rendered)
        removed_any = removed_any or removed

    if removed_any:
        lines.insert(0, REMOVAL_HEADER_COMMENT)

    return "\n".join(lines).rstrip() + "\n"


# --- CLI ----------------------------------------------------------------------


def read_clipboard() -> str:
    return subprocess.check_output(["pbpaste"], text=True)


def write_clipboard(s: str) -> None:
    subprocess.run(["pbcopy"], input=s, text=True, check=True)


def read_input(args: argparse.Namespace) -> str:
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            return f.read()
    if args.pbpaste or sys.stdin.isatty():
        try:
            return read_clipboard()
        except Exception:
            # Fall back to stdin if clipboard is unavailable.
            pass
    return sys_stdin_read()


def sys_stdin_read() -> str:
    return sys.stdin.read()


def write_output(args: argparse.Namespace, s: str) -> str:
    """
    Write the simplified HTML somewhere and report where it went.
    Returns a short destination string for logging.
    """
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(s)
        return f'file "{args.output}"'

    if sys.stdout.isatty():
        try:
            write_clipboard(s)
            return "clipboard"
        except Exception:
            # Fall back to stdout if clipboard is unavailable.
            print(s, end="")
            return "stdout"

    print(s, end="")
    return "stdout"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Condense DevTools outerHTML into LLM-readable HTML. Defaults to clipboard -> clipboard when run interactively.")
    ap.add_argument("--in", dest="input", help="Input file (defaults to stdin).")
    ap.add_argument("--out", dest="output", help="Output file (defaults to stdout).")
    ap.add_argument("--pbpaste", action="store_true", help="Read input from macOS clipboard (default when stdin is a TTY).")

    ap.add_argument("--max-depth", type=int, default=20, help="Max DOM depth before summarizing.")
    ap.add_argument("--max-children", type=int, default=60, help="Max children per node before summarizing.")
    ap.add_argument("--max-attr-len", type=int, default=140, help="Max length per attribute value.")
    ap.add_argument("--max-text-len", type=int, default=220, help="Max length per text node.")
    ap.add_argument("--max-classes", type=int, default=10, help="Max classes to keep in class attribute.")

    ap.add_argument("--keep-style", action="store_true", help="Keep style attributes (still truncated).")
    ap.add_argument("--keep-data-attrs", action="store_true", help="Keep ALL data-* attributes (still truncated).")
    ap.add_argument("--indent", default="  ", help="Indent string (default: two spaces).")
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    raw = read_input(args)
    out = simplify_html(
        raw,
        max_depth=args.max_depth,
        max_children=args.max_children,
        max_attr_len=args.max_attr_len,
        max_text_len=args.max_text_len,
        max_classes=args.max_classes,
        keep_style=args.keep_style,
        keep_data_attrs=args.keep_data_attrs,
        indent=args.indent,
    )
    dest = write_output(args, out)
    sys.stderr.write(f"[ok] Simplified HTML written to {dest}\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
