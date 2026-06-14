"""
Page 4 - Association Rule Network Graph

Interactive NetworkX + Plotly graph where:
  • Nodes  = products
  • Edges  = association rules
  • Edge colour = lift tier (green > 3, yellow 2-3, red < 2)
  • Edge hover  = support, confidence, lift
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))
from rule_filtering import filter_rules
import theme

st.set_page_config(page_title="Network Graph · SmartBasket AI", page_icon="🔗", layout="wide")
theme.apply_theme()


# Data
@st.cache_data(show_spinner=False)
def load_rules():
    return pd.read_csv(BASE / "models" / "association_rules.csv")

with st.spinner("Loading rules…"):
    rules_raw = load_rules()


# Masthead
theme.masthead(
    eyebrow="Rule Network",
    title="Association Rule Network",
    subtitle="Products are nodes, association rules are edges, and edge colour reflects lift "
             "strength. Larger nodes are more connected.",
)


# Sidebar controls
with st.sidebar:
    st.markdown("##### Graph controls")

    graph_min_lift = st.slider("Min lift", 1.0, 15.0, 3.0, 0.5,
                                help="Only show rules with lift ≥ this value")
    graph_min_conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
    max_nodes = st.slider("Max nodes", 10, 80, 35,
                           help="Limit the graph size for readability")
    max_edges = st.slider("Max edges", 10, 200, 80)

    search_node = st.text_input("🔎 Highlight product",
                                placeholder="e.g. JUMBO BAG",
                                help="Products matching this text will be highlighted in orange")

    st.markdown("---")
    st.markdown("##### Edge colour legend")
    st.markdown('<div class="legend-row"><span class="legend-dot" style="background:#10b981"></span>Lift &gt; 3 (strong)</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend-row"><span class="legend-dot" style="background:#f59e0b"></span>Lift 2-3 (moderate)</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend-row"><span class="legend-dot" style="background:#ef4444"></span>Lift &lt; 2 (weak)</div>', unsafe_allow_html=True)


# Filter rules for graph
graph_rules = filter_rules(
    rules_raw,
    min_support=0.001,
    min_confidence=graph_min_conf,
    min_lift=graph_min_lift,
)
graph_rules = graph_rules.sort_values("lift", ascending=False).head(max_edges)


def _edge_colour(lift: float) -> str:
    if lift >= 3.0:
        return "#10b981"
    if lift >= 2.0:
        return "#f59e0b"
    return "#ef4444"


# Build NetworkX graph
G = nx.DiGraph()
for _, row in graph_rules.iterrows():
    ant = row["antecedents_str"].strip()
    con = row["consequents_str"].strip()
    G.add_edge(
        ant, con,
        weight=float(row["lift"]),
        confidence=float(row["confidence"]),
        support=float(row["support"]),
    )

# Limit nodes (keep highest-degree ones)
if G.number_of_nodes() > max_nodes:
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
    keep = {n for n, _ in top_nodes}
    G = G.subgraph(keep).copy()

if G.number_of_nodes() == 0:
    st.warning("No rules match the current filters. Try lowering the Min Lift or Min Confidence.")
    st.stop()


# Layout
pos = nx.spring_layout(G, k=2.5, seed=42, iterations=50)


# Plotly traces
search_upper = search_node.strip().upper()

# Edge line traces (grouped by lift tier for legend) + midpoint hover markers
edge_traces = {}
mid_x, mid_y, mid_text = [], [], []

for u, v, data in G.edges(data=True):
    colour = _edge_colour(data["weight"])
    tier = (
        "Lift > 3"  if data["weight"] >= 3.0 else
        "Lift 2-3"  if data["weight"] >= 2.0 else
        "Lift < 2"
    )
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    tip = (
        f"<b>{u}</b> → <b>{v}</b><br>"
        f"Lift: {data['weight']:.2f}×<br>"
        f"Confidence: {data['confidence']:.2%}<br>"
        f"Support: {data['support']:.4f}"
    )
    if tier not in edge_traces:
        edge_traces[tier] = {"x": [], "y": [], "colour": colour}
    edge_traces[tier]["x"] += [x0, x1, None]
    edge_traces[tier]["y"] += [y0, y1, None]

    # Midpoint marker for hover tooltip
    mid_x.append((x0 + x1) / 2)
    mid_y.append((y0 + y1) / 2)
    mid_text.append(tip)

fig = go.Figure()

tier_colours = {"Lift > 3": "#10b981", "Lift 2-3": "#f59e0b", "Lift < 2": "#ef4444"}
for tier, data in edge_traces.items():
    fig.add_trace(go.Scatter(
        x=data["x"], y=data["y"],
        mode="lines",
        line=dict(color=tier_colours[tier], width=1.8),
        name=tier,
        hoverinfo="skip",
        showlegend=True,
    ))

# Invisible midpoint markers - carry the edge hover tooltips
fig.add_trace(go.Scatter(
    x=mid_x, y=mid_y,
    mode="markers",
    marker=dict(size=10, opacity=0, color="rgba(0,0,0,0)"),
    hovertext=mid_text,
    hoverinfo="text",
    hoverlabel=dict(bgcolor="white", bordercolor=theme.BORDER,
                    font=dict(size=12, color=theme.INK)),
    showlegend=False,
    name="",
))

# Node trace
node_x, node_y, node_text, node_hover, node_color, node_size = [], [], [], [], [], []
degree_map = dict(G.degree())
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node[:22] + "…" if len(node) > 22 else node)
    degree = degree_map[node]
    node_hover.append(f"<b>{node}</b><br>Connections: {degree}")
    highlight = search_upper and search_upper in node.upper()
    node_color.append("#f97316" if highlight else theme.ACCENT)
    node_size.append(14 + degree * 2)

fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=node_text,
    textposition="top center",
    textfont=dict(size=9, color=theme.INK_2, family="Hanken Grotesk, sans-serif"),
    hovertext=node_hover,
    hoverinfo="text",
    marker=dict(
        size=node_size,
        color=node_color,
        line=dict(width=1.5, color="white"),
    ),
    name="Product",
    showlegend=False,
))

fig.update_layout(
    height=620,
    showlegend=True,
    font=dict(family="Hanken Grotesk, sans-serif", color=theme.INK_2),
    legend=dict(
        orientation="h", y=-0.07, x=0.5, xanchor="center",
        font=dict(size=12),
    ),
    hovermode="closest",
    margin=dict(t=20, b=60, l=10, r=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)

# Stats strip
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
theme.chips([
    f"<b>{n_nodes}</b> products",
    f"<b>{n_edges}</b> rule edges",
    f"min lift <b>{graph_min_lift}×</b>",
])
st.markdown("")

with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True)


# Edge table
with st.expander(f"📋 View edge list ({n_edges} rules)"):
    edge_rows = []
    for u, v, d in G.edges(data=True):
        edge_rows.append({
            "From": u, "To": v,
            "Lift": round(d["weight"], 3),
            "Confidence": f"{d['confidence']:.2%}",
            "Support": f"{d['support']:.4f}",
        })
    edge_df = pd.DataFrame(edge_rows).sort_values("Lift", ascending=False)
    edge_df.index = range(1, len(edge_df) + 1)
    st.dataframe(edge_df, use_container_width=True, height=300)
