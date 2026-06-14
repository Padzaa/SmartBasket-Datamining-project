"""
SmartBasket AI — shared design system.

A single source of truth for the look & feel of every page. Import once per
page and call `apply_theme()` to inject the global stylesheet, then use the
small render helpers (`masthead`, `section`, `kpi_row`, `nav_card`, `style_fig`)
so every page speaks the same visual language.

Design language: light "premium SaaS" — warm off-white canvas, crisp white
surfaces with hairline borders and soft shadows, a confident teal accent used
sparingly, and a typographic system built on Bricolage Grotesque (display),
Hanken Grotesk (body) and JetBrains Mono (labels / data).
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence

import streamlit as st

# ── Design tokens ───────────────────────────────────────────────────────────
INK        = "#0d1321"   # primary text
INK_2      = "#475569"   # secondary text
MUTED      = "#94a3b8"   # tertiary / captions
CANVAS     = "#f6f7f9"   # page background
SURFACE    = "#ffffff"   # cards
BORDER     = "#e8eaee"   # hairline borders
ACCENT     = "#0d9488"   # teal — primary accent
ACCENT_DK  = "#0f766e"   # teal — deep
ACCENT_SOFT= "#f0fdfa"   # teal — tint
INDIGO     = "#6366f1"   # secondary accent (AI / ALS signal)

# Sequential teal scale reused by charts so data viz stays on-brand.
TEAL_SCALE = ["#cffafe", "#5eead4", "#0d9488", "#115e59"]

FONT_DISPLAY = "'Bricolage Grotesque', 'Hanken Grotesk', sans-serif"
FONT_BODY    = "'Hanken Grotesk', system-ui, sans-serif"
FONT_MONO    = "'JetBrains Mono', ui-monospace, monospace"


# ── Global stylesheet ────────────────────────────────────────────────────────
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700;12..96,800&family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [class*="st-"],
.stMarkdown, .stButton, button, input, textarea, select {{
    font-family: {FONT_BODY};
}}
/* Restore Streamlit's icon font (the broad rule above would otherwise turn
   icon ligatures like the sidebar collapse arrow into literal text). */
[data-testid="stIconMaterial"], .material-icons,
[class*="material-symbols"], .material-symbols-rounded {{
    font-family: 'Material Symbols Rounded' !important;
}}
[data-testid="stAppViewContainer"] {{
    background: {CANVAS};
    background-image:
        radial-gradient(900px 500px at 100% -5%, rgba(13,148,136,0.06), transparent 60%),
        radial-gradient(700px 420px at -5% 0%, rgba(99,102,241,0.05), transparent 55%);
}}
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1180px; }}

h1, h2, h3, h4 {{ font-family: {FONT_DISPLAY}; color: {INK}; letter-spacing: -0.02em; }}

/* Tame Streamlit's default top toolbar / decoration */
[data-testid="stDecoration"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
[data-testid="stSidebarNav"] {{ font-family: {FONT_BODY}; }}
[data-testid="stSidebarNav"] a span {{ font-size: 0.95rem; font-weight: 500; }}

/* ── Masthead (page header) ───────────────────────────────────────────── */
.sb-masthead {{ margin: 0 0 1.9rem; }}
.sb-eyebrow {{
    font-family: {FONT_MONO};
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.18em;
    text-transform: uppercase; color: {ACCENT_DK};
    display: inline-flex; align-items: center; gap: 0.5rem;
}}
.sb-eyebrow::before {{
    content: ""; width: 22px; height: 2px; background: {ACCENT};
    display: inline-block; border-radius: 2px;
}}
.sb-masthead h1 {{
    font-size: 2.55rem; font-weight: 700; margin: 0.55rem 0 0.4rem; line-height: 1.05;
}}
.sb-masthead .sb-sub {{
    font-size: 1.02rem; color: {INK_2}; margin: 0; max-width: 60ch;
}}
.sb-rule {{ height: 1px; background: {BORDER}; margin: 1.4rem 0 0; }}

/* ── Section header ───────────────────────────────────────────────────── */
.sb-section {{ margin: 2.2rem 0 1.1rem; }}
.sb-section .lbl {{
    font-family: {FONT_DISPLAY}; font-size: 1.18rem; font-weight: 600;
    color: {INK}; display: flex; align-items: center; gap: 0.6rem;
}}
.sb-section .lbl::before {{
    content: ""; width: 6px; height: 18px; border-radius: 3px;
    background: linear-gradient(180deg, {ACCENT}, {ACCENT_DK});
}}
.sb-section .sub {{ font-size: 0.88rem; color: {MUTED}; margin: 0.25rem 0 0 1.2rem; }}

/* ── KPI cards ────────────────────────────────────────────────────────── */
.sb-kpi-row {{ display: flex; gap: 0.9rem; flex-wrap: wrap; }}
.sb-kpi {{
    flex: 1 1 0; min-width: 150px;
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: 1.15rem 1.25rem; position: relative; overflow: hidden;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.05);
    transition: transform .18s ease, box-shadow .18s ease;
}}
.sb-kpi:hover {{ transform: translateY(-2px); box-shadow: 0 10px 24px rgba(16,24,40,0.08); }}
.sb-kpi::after {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: var(--accent, {ACCENT});
}}
.sb-kpi .k-lbl {{
    font-family: {FONT_MONO}; font-size: 0.68rem; letter-spacing: 0.13em;
    text-transform: uppercase; color: {MUTED};
}}
.sb-kpi .k-val {{
    font-family: {FONT_DISPLAY}; font-size: 1.95rem; font-weight: 700;
    color: {INK}; line-height: 1.1; margin-top: 0.35rem;
    font-feature-settings: "tnum" 1;
}}
.sb-kpi .k-sub {{ font-size: 0.78rem; color: {INK_2}; margin-top: 0.25rem; }}

/* ── Nav cards (home) ─────────────────────────────────────────────────── */
.sb-nav {{
    display: block; background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 18px; padding: 1.5rem 1.6rem; height: 100%;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}}
.sb-nav:hover {{
    transform: translateY(-3px); border-color: #d4f3ee;
    box-shadow: 0 14px 30px rgba(13,148,136,0.10);
}}
.sb-nav .n-top {{ display: flex; align-items: center; justify-content: space-between; }}
.sb-nav .n-icon {{
    width: 44px; height: 44px; border-radius: 12px;
    background: {ACCENT_SOFT}; color: {ACCENT_DK};
    display: flex; align-items: center; justify-content: center; font-size: 1.35rem;
}}
.sb-nav .n-arrow {{ color: {MUTED}; font-size: 1.2rem; transition: transform .18s ease; }}
.sb-nav:hover .n-arrow {{ transform: translate(3px,-3px); color: {ACCENT}; }}
.sb-nav h3 {{ font-size: 1.12rem; font-weight: 700; margin: 1rem 0 0.3rem; }}
.sb-nav p  {{ font-size: 0.88rem; color: {INK_2}; margin: 0; line-height: 1.5; }}

/* ── Chips / pills ────────────────────────────────────────────────────── */
.sb-chip {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 999px;
    padding: 0.32rem 0.85rem; font-size: 0.82rem; color: {INK_2};
    margin: 0.2rem 0.35rem 0.2rem 0; font-weight: 500;
}}
.sb-chip b {{ color: {INK}; font-weight: 700; }}
.sb-chip .dot {{ width: 7px; height: 7px; border-radius: 50%; background: {ACCENT}; }}

/* ── Sidebar mini-stats ───────────────────────────────────────────────── */
.sb-side-stat {{
    display: flex; align-items: baseline; justify-content: space-between;
    padding: 0.5rem 0; border-bottom: 1px dashed {BORDER};
}}
.sb-side-stat .s-lbl {{ font-size: 0.82rem; color: {INK_2}; }}
.sb-side-stat .s-val {{ font-family: {FONT_DISPLAY}; font-weight: 700; color: {INK}; }}

/* ── Recommendation cards ─────────────────────────────────────────────── */
.rec-card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: 1.15rem 1.3rem; margin-bottom: 0.85rem; position: relative;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    transition: box-shadow .18s ease, transform .18s ease;
}}
.rec-card:hover {{ box-shadow: 0 10px 26px rgba(16,24,40,0.08); transform: translateY(-1px); }}
.rec-card .rank-badge {{
    position: absolute; top: 1.1rem; right: 1.2rem;
    font-family: {FONT_MONO}; font-size: 0.72rem; font-weight: 600;
    color: {ACCENT_DK}; background: {ACCENT_SOFT};
    border: 1px solid #cdeee8; border-radius: 8px; padding: 0.12rem 0.5rem;
}}
.rec-card h4 {{ font-size: 1.04rem; font-weight: 700; color: {INK}; margin: 0 2.5rem 0.4rem 0; }}
.rec-card .explanation {{ font-size: 0.88rem; color: {INK_2}; margin-bottom: 0.7rem; line-height: 1.5; }}
.score-bar {{ height: 6px; border-radius: 3px; background: #eef0f3; margin: 0.2rem 0 0.7rem; }}
.score-fill {{ height: 6px; border-radius: 3px; background: linear-gradient(90deg, {ACCENT}, {INDIGO}); }}
.meta-pill {{
    display: inline-block; font-family: {FONT_MONO}; font-size: 0.72rem;
    background: #f8fafc; border: 1px solid {BORDER}; border-radius: 7px;
    padding: 0.16rem 0.55rem; color: {INK_2}; margin-right: 0.35rem;
}}
.no-rec-box {{
    background: #fffbeb; border: 1px solid #fde68a; border-left: 3px solid #f59e0b;
    border-radius: 12px; padding: 1rem 1.2rem; color: #92400e; font-size: 0.9rem;
}}
.fallback-item {{ padding: 0.55rem 0; border-bottom: 1px solid {BORDER}; font-size: 0.92rem; color: {INK_2}; }}

/* ── Toolbar / count badge ────────────────────────────────────────────── */
.sb-count {{
    font-family: {FONT_MONO}; font-size: 0.84rem; color: {ACCENT_DK};
    background: {ACCENT_SOFT}; border: 1px solid #cdeee8; border-radius: 10px;
    padding: 0.4rem 0.85rem; display: inline-block; font-weight: 600;
}}
.sb-hint {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.35rem; }}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {{
    background: {ACCENT}; border: 1px solid {ACCENT_DK}; border-radius: 11px;
    font-weight: 600; box-shadow: 0 1px 2px rgba(13,148,136,0.25);
}}
.stButton > button[kind="primary"]:hover {{ background: {ACCENT_DK}; }}

/* Generic surface card wrapper via st.container(border=True) */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 16px !important; border-color: {BORDER} !important;
    background: {SURFACE}; box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}}

/* Legend rows for the network graph */
.legend-row {{ font-size: 0.86rem; color: {INK_2}; margin: 0.35rem 0; display: flex; align-items: center; }}
.legend-dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 8px; }}
</style>
"""


def apply_theme() -> None:
    """Inject the global stylesheet. Call once at the top of every page."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Render helpers ────────────────────────────────────────────────────────--
def masthead(eyebrow: str, title: str, subtitle: str = "") -> None:
    """Page header: mono eyebrow, large display title, subtitle, hairline."""
    sub = f'<p class="sb-sub">{html.escape(subtitle)}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="sb-masthead">
          <span class="sb-eyebrow">{html.escape(eyebrow)}</span>
          <h1>{html.escape(title)}</h1>
          {sub}
          <div class="sb-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(label: str, sub: str = "") -> None:
    """Section header with an accent tick."""
    sub_html = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
    st.markdown(
        f'<div class="sb-section"><div class="lbl">{html.escape(label)}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: Sequence[dict]) -> None:
    """Render a row of KPI cards.

    Each item: {"label": str, "value": str, "sub": str?, "accent": str?}
    """
    cards = []
    for it in items:
        accent = it.get("accent", ACCENT)
        sub = f'<div class="k-sub">{html.escape(str(it["sub"]))}</div>' if it.get("sub") else ""
        cards.append(
            f'<div class="sb-kpi" style="--accent:{accent}">'
            f'<div class="k-lbl">{html.escape(str(it["label"]))}</div>'
            f'<div class="k-val">{html.escape(str(it["value"]))}</div>'
            f'{sub}</div>'
        )
    st.markdown(f'<div class="sb-kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def nav_card(icon: str, title: str, desc: str) -> str:
    """Return HTML for a navigation card (home page)."""
    return (
        f'<div class="sb-nav">'
        f'<div class="n-top"><div class="n-icon">{icon}</div>'
        f'<div class="n-arrow">↗</div></div>'
        f'<h3>{html.escape(title)}</h3>'
        f'<p>{html.escape(desc)}</p></div>'
    )


def chips(items: Iterable[str]) -> None:
    """Render a row of chips (used for quick insights)."""
    html_chips = "".join(
        f'<span class="sb-chip"><span class="dot"></span>{c}</span>' for c in items
    )
    st.markdown(html_chips, unsafe_allow_html=True)


def style_fig(fig, height: int | None = None):
    """Apply the house style to a Plotly figure: transparent canvas, brand
    fonts, restrained gridlines. Traces and colours are left untouched so the
    existing (good) charts keep their identity."""
    # Preserve any existing title text — passing a title dict without `text`
    # makes Plotly render the literal string "undefined" as the chart title.
    fig.update_layout(
        font=dict(family="Hanken Grotesk, sans-serif", color=INK_2, size=12),
        title=dict(text=fig.layout.title.text,
                   font=dict(family="Bricolage Grotesque, sans-serif",
                             color=INK, size=15), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=44, b=12),
        xaxis=dict(showgrid=False, zerolinecolor=BORDER),
        yaxis=dict(gridcolor="#f0f1f4", zerolinecolor=BORDER),
        legend=dict(font=dict(size=11)),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig
