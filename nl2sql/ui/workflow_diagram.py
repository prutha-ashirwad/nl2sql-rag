"""An SVG rendering of the compiled LangGraph workflow.

Edges are transcribed from :func:`nl2sql.graph.workflow.build_workflow` and its
routers; :data:`EDGES` is the one place to update when a route changes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from nl2sql.graph.nodes import (
    NODE_ANALYZE,
    NODE_EXECUTE,
    NODE_FINALIZE,
    NODE_GENERATE,
    NODE_REPAIR,
    NODE_RETRIEVE,
    NODE_VALIDATE,
)

# --- Palette ---------------------------------------------------------------
# Matched to .streamlit/config.toml.
_SURFACE = "#161C27"
_BORDER = "#2A3341"
_TEXT = "#E4E9F2"
_MUTED = "#94A3B8"
_FAINT = "#64748B"
_ACCENT = "#6D8CFF"  # retrieval / the model path
_GUARD = "#FBBF24"  # validation and the repair loop
_RUN = "#34D399"  # execution
_DIM = 0.34  # opacity for nodes this run did not visit

# --- Geometry --------------------------------------------------------------
# Font sizes run large because the SVG is scaled down to the container width.
_BOX_W = 138
_BOX_H = 64
_GAP = 64
_SPINE_Y = 132
_REPAIR_Y = 26
_RAIL_Y = 274
_LEFT = 52

_FONT_TITLE = 17
_FONT_SUB = 12
_FONT_EDGE = 11.5


@dataclass(frozen=True, slots=True)
class _Node:
    """One box in the diagram."""

    key: str
    title: str
    subtitle: str
    accent: str
    x: float
    y: float


# The main path, left to right, in the order the graph runs them.
_SPINE = (
    (NODE_ANALYZE, "analyze", "classify · resolve", _ACCENT),
    (NODE_RETRIEVE, "retrieve", "RAG over the KB", _ACCENT),
    (NODE_GENERATE, "generate", "write the SQL", _ACCENT),
    (NODE_VALIDATE, "validate", "check vs the KB", _GUARD),
    (NODE_EXECUTE, "execute", "read-only, capped", _RUN),
    (NODE_FINALIZE, "finalize", "compose the answer", _MUTED),
)

NODES: dict[str, _Node] = {
    key: _Node(key, title, subtitle, accent, _LEFT + index * (_BOX_W + _GAP), _SPINE_Y)
    for index, (key, title, subtitle, accent) in enumerate(_SPINE)
}
# The repair loop sits above the spine so the early-exit rail below stays unobstructed.
NODES[NODE_REPAIR] = _Node(
    NODE_REPAIR,
    "repair",
    "feed errors back",
    _GUARD,
    NODES[NODE_VALIDATE].x,
    _REPAIR_Y,
)

_WIDTH = _LEFT + 6 * (_BOX_W + _GAP) + 40
_HEIGHT = _RAIL_Y + 54

# Every conditional route in the graph, with the condition that selects it.
EDGES: tuple[tuple[str, str, str], ...] = (
    (NODE_ANALYZE, NODE_RETRIEVE, "question is supported"),
    (NODE_RETRIEVE, NODE_GENERATE, "schema was found"),
    (NODE_GENERATE, NODE_VALIDATE, "a query was produced"),
    (NODE_VALIDATE, NODE_EXECUTE, "valid, and a database is configured"),
    (NODE_VALIDATE, NODE_REPAIR, "invalid, and the repair budget has room"),
    (NODE_REPAIR, NODE_VALIDATE, "always — re-validate the rewrite"),
    (NODE_EXECUTE, NODE_FINALIZE, "always"),
    (NODE_ANALYZE, NODE_FINALIZE, "the question cannot be served"),
    (NODE_RETRIEVE, NODE_FINALIZE, "nothing in the KB matched"),
    (NODE_GENERATE, NODE_FINALIZE, "no query was produced"),
    (NODE_VALIDATE, NODE_FINALIZE, "valid but execution is off, or the budget is spent"),
)


def _centre_x(node: _Node) -> float:
    return node.x + _BOX_W / 2


def _box(node: _Node, *, active: bool) -> str:
    """Draw one node."""
    opacity = 1.0 if active else _DIM
    stroke = node.accent if active else _BORDER
    title_fill = _TEXT if active else _MUTED
    return f"""
  <g opacity="{opacity}">
    <rect x="{node.x}" y="{node.y}" width="{_BOX_W}" height="{_BOX_H}" rx="10"
          fill="{_SURFACE}" stroke="{stroke}" stroke-width="1.25"/>
    <rect x="{node.x}" y="{node.y + 10}" width="3" height="{_BOX_H - 20}" rx="1.5"
          fill="{node.accent}"/>
    <text x="{node.x + 15}" y="{node.y + 27}" fill="{title_fill}"
          font-size="{_FONT_TITLE}" font-weight="600"
          font-family="inherit">{node.title}</text>
    <text x="{node.x + 15}" y="{node.y + 47}" fill="{_FAINT}"
          font-size="{_FONT_SUB}" font-family="inherit">{node.subtitle}</text>
  </g>"""


def _spine_edge(left: _Node, right: _Node, label: str, *, active: bool) -> str:
    """Draw a straight left-to-right edge on the main path."""
    colour = _MUTED if active else _BORDER
    marker = "arrow" if active else "arrow-dim"
    start = left.x + _BOX_W
    end = right.x - 7
    mid = (start + end) / 2
    y = _SPINE_Y + _BOX_H / 2
    return f"""
  <g opacity="{1.0 if active else _DIM}">
    <line x1="{start}" y1="{y}" x2="{end}" y2="{y}" stroke="{colour}"
          stroke-width="1.5" marker-end="url(#{marker})"/>
    <text x="{mid}" y="{y - 10}" fill="{_FAINT}" font-size="{_FONT_EDGE}"
          text-anchor="middle" font-family="inherit">{label}</text>
  </g>"""


def render(visited: Iterable[str] | None = None) -> str:
    """Return the workflow as a standalone SVG.

    Args:
        visited: Node names the run being shown entered; unvisited nodes are dimmed.
            ``None`` draws every node at full strength.
    """
    seen = set(visited) if visited is not None else set(NODES)
    live = {key: key in seen for key in NODES}

    validate = NODES[NODE_VALIDATE]
    finalize = NODES[NODE_FINALIZE]

    parts: list[str] = []

    # Main path.
    for left_key, right_key, label in (
        # Abbreviated to fit one gap; the full conditions are in EDGES.
        (NODE_ANALYZE, NODE_RETRIEVE, "supported"),
        (NODE_RETRIEVE, NODE_GENERATE, "matched"),
        (NODE_GENERATE, NODE_VALIDATE, "has SQL"),
        (NODE_VALIDATE, NODE_EXECUTE, "valid"),
        (NODE_EXECUTE, NODE_FINALIZE, ""),
    ):
        parts.append(
            _spine_edge(
                NODES[left_key],
                NODES[right_key],
                label,
                active=live[left_key] and live[right_key],
            )
        )

    # The bounded repair loop: up into repair on a failed check, back down to re-check.
    loop_active = live[NODE_REPAIR]
    loop_colour = _GUARD if loop_active else _BORDER
    loop_marker = "arrow-guard" if loop_active else "arrow-dim"
    up_x = validate.x + 38
    down_x = validate.x + _BOX_W - 38
    label_y = (_REPAIR_Y + _BOX_H + _SPINE_Y) / 2 + 4
    parts.append(f"""
  <g opacity="{1.0 if loop_active else _DIM}">
    <line x1="{up_x}" y1="{_SPINE_Y}" x2="{up_x}" y2="{_REPAIR_Y + _BOX_H + 7}"
          stroke="{loop_colour}" stroke-width="1.5"
          marker-end="url(#{loop_marker})"/>
    <text x="{up_x - 9}" y="{label_y}" fill="{_FAINT}" font-size="{_FONT_EDGE}"
          text-anchor="end" font-family="inherit">invalid</text>
    <line x1="{down_x}" y1="{_REPAIR_Y + _BOX_H}" x2="{down_x}" y2="{_SPINE_Y - 7}"
          stroke="{loop_colour}" stroke-width="1.5"
          marker-end="url(#{loop_marker})"/>
    <text x="{down_x + 9}" y="{label_y}" fill="{_FAINT}" font-size="{_FONT_EDGE}"
          font-family="inherit">re-check</text>
  </g>""")

    # Early exits, every one of them landing on finalize.
    rail_from = _centre_x(NODES[NODE_ANALYZE])
    rail_to = _centre_x(finalize)
    spine_bottom = _SPINE_Y + _BOX_H
    rail = (
        f"M {rail_from} {spine_bottom} V {_RAIL_Y} "
        f"H {rail_to} V {spine_bottom + 7}"
    )
    parts.append(f"""
  <g opacity="0.75">
    <path d="{rail}"
          fill="none" stroke="{_BORDER}" stroke-width="1.4" stroke-dasharray="5 4"
          marker-end="url(#arrow-dim)"/>""")
    for key in (NODE_RETRIEVE, NODE_GENERATE, NODE_VALIDATE):
        x = _centre_x(NODES[key])
        parts.append(
            f"""
    <line x1="{x}" y1="{_SPINE_Y + _BOX_H}" x2="{x}" y2="{_RAIL_Y}"
          stroke="{_BORDER}" stroke-width="1.4" stroke-dasharray="5 4"/>"""
        )
    rail_label = (
        "early exit — unsupported question · nothing retrieved · no query produced · "
        "repair budget spent · execution off"
    )
    parts.append(f"""
    <text x="{rail_from + 12}" y="{_RAIL_Y - 11}" fill="{_FAINT}"
          font-size="{_FONT_EDGE}" font-family="inherit">{rail_label}</text>
  </g>""")

    # START / END terminals.
    analyze = NODES[NODE_ANALYZE]
    y = _SPINE_Y + _BOX_H / 2
    parts.append(f"""
  <g>
    <text x="2" y="{y + 4}" fill="{_FAINT}" font-size="{_FONT_EDGE}"
          font-weight="700" letter-spacing="0.08em"
          font-family="inherit">START</text>
    <line x1="{_LEFT - 8}" y1="{y}" x2="{analyze.x - 7}" y2="{y}" stroke="{_MUTED}"
          stroke-width="1.5" marker-end="url(#arrow)"/>
    <line x1="{finalize.x + _BOX_W}" y1="{y}" x2="{finalize.x + _BOX_W + 22}" y2="{y}"
          stroke="{_MUTED}" stroke-width="1.5" marker-end="url(#arrow)"/>
    <text x="{finalize.x + _BOX_W + 28}" y="{y + 4}" fill="{_FAINT}"
          font-size="{_FONT_EDGE}" font-weight="700" letter-spacing="0.08em"
          font-family="inherit">END</text>
  </g>""")

    for key, node in NODES.items():
        parts.append(_box(node, active=live[key]))

    markers = "".join(
        f"""
    <marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
            markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{colour}"/>
    </marker>"""
        for name, colour in (
            ("arrow", _MUTED),
            ("arrow-dim", _BORDER),
            ("arrow-guard", _GUARD),
        )
    )

    return f"""<svg viewBox="0 0 {_WIDTH} {_HEIGHT}" width="100%"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="The NL2SQL LangGraph workflow"
     style="max-width:100%;height:auto;font-family:inherit">
  <defs>{markers}
  </defs>{"".join(parts)}
</svg>"""
