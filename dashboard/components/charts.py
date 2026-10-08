"""Interactive Plotly chart components configured for enterprise light theme."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Design System Palette Tokens
TEXT_PRIMARY = "#172033"
TEXT_SECONDARY = "#5B6577"
GRID_COLOR = "#E2E8F0"
RISK_COLORS = {
    "CRITICAL": "#B91C1C",
    "HIGH": "#C2410C",
    "MEDIUM": "#B45309",
    "LOW": "#15803D",
}


def plot_risk_distribution(counts_dict: dict) -> go.Figure:
    """Plot risk category distribution donut chart with clean light styling."""
    labels = list(counts_dict.keys())
    values = list(counts_dict.values())
    color_seq = [RISK_COLORS.get(lvl, "#64748B") for lvl in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker={"colors": color_seq, "line": {"color": "#FFFFFF", "width": 2}},
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont={"color": "#FFFFFF", "size": 12},
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_PRIMARY, "family": "Inter, sans-serif"},
        margin={"t": 16, "b": 16, "l": 16, "r": 16},
        showlegend=False,
    )
    return fig


def plot_churn_by_contract(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing observed churn rate across contract types."""
    if "Contract" not in df.columns or "Churn" not in df.columns:
        return go.Figure()

    churn_by_contract = df.groupby("Contract")["Churn"].mean().reset_index()
    churn_by_contract["ChurnRatePct"] = churn_by_contract["Churn"] * 100

    fig = px.bar(
        churn_by_contract,
        x="Contract",
        y="ChurnRatePct",
        text_auto=".1f",
        labels={"ChurnRatePct": "Observed Churn (%)", "Contract": "Contract Type"},
        color_discrete_sequence=["#2563EB"],
    )
    fig.update_traces(
        marker_color="#2563EB",
        marker_line_color="#1D4ED8",
        marker_line_width=1,
        textfont={"color": "#FFFFFF", "size": 13, "family": "Inter, sans-serif"},
        textposition="inside",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_PRIMARY, "family": "Inter, sans-serif"},
        xaxis={
            "showgrid": False,
            "title": {"font": {"color": TEXT_SECONDARY, "size": 13}},
            "tickfont": {"color": TEXT_PRIMARY, "size": 12},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "title": {"font": {"color": TEXT_SECONDARY, "size": 13}},
            "tickfont": {"color": TEXT_SECONDARY, "size": 12},
        },
        margin={"t": 24, "b": 16, "l": 16, "r": 16},
    )
    return fig


def plot_churn_by_tenure_bucket(df: pd.DataFrame) -> go.Figure:
    """Bar chart of observed churn across tenure cohorts."""
    if "tenure" not in df.columns or "Churn" not in df.columns:
        return go.Figure()

    df_copy = df.copy()
    bins = [-1, 12, 24, 48, 60, 72]
    labels = ["0-12m", "12-24m", "24-48m", "48-60m", "60-72m"]
    df_copy["tenure_bucket"] = pd.cut(df_copy["tenure"], bins=bins, labels=labels)
    grouped = df_copy.groupby("tenure_bucket", observed=False)["Churn"].mean().reset_index()
    grouped["ChurnRatePct"] = grouped["Churn"] * 100

    fig = px.bar(
        grouped,
        x="tenure_bucket",
        y="ChurnRatePct",
        text_auto=".1f",
        labels={"ChurnRatePct": "Observed Churn (%)", "tenure_bucket": "Tenure Cohort"},
        color_discrete_sequence=["#3B82F6"],
    )
    fig.update_traces(
        marker_color="#3B82F6",
        marker_line_color="#2563EB",
        marker_line_width=1,
        textfont={"color": "#FFFFFF", "size": 13, "family": "Inter, sans-serif"},
        textposition="inside",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_PRIMARY, "family": "Inter, sans-serif"},
        xaxis={
            "showgrid": False,
            "title": {"font": {"color": TEXT_SECONDARY, "size": 13}},
            "tickfont": {"color": TEXT_PRIMARY, "size": 12},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "title": {"font": {"color": TEXT_SECONDARY, "size": 13}},
            "tickfont": {"color": TEXT_SECONDARY, "size": 12},
        },
        margin={"t": 24, "b": 16, "l": 16, "r": 16},
    )
    return fig


def plot_priority_scatter(df_results: pd.DataFrame) -> go.Figure:
    """Scatter plot: Churn Probability vs Customer Lifetime Value sized by Priority Score."""
    fig = px.scatter(
        df_results,
        x="churn_probability",
        y="clv",
        size="retention_priority_score",
        color="risk_level",
        hover_data=["customer_id", "retention_priority_score"],
        color_discrete_map=RISK_COLORS,
        labels={
            "churn_probability": "Estimated Churn Probability",
            "clv": "Customer Lifetime Value ($)",
            "risk_level": "Risk Level",
        },
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_PRIMARY, "family": "Inter, sans-serif"},
        xaxis={
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "title": {"font": {"color": TEXT_SECONDARY, "size": 13}},
            "tickfont": {"color": TEXT_SECONDARY, "size": 12},
            "tickformat": ".0%",
        },
        yaxis={
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "title": {"font": {"color": TEXT_SECONDARY, "size": 13}},
            "tickfont": {"color": TEXT_SECONDARY, "size": 12},
            "tickprefix": "$",
        },
        margin={"t": 16, "b": 16, "l": 16, "r": 16},
        legend={
            "title": {"text": "Risk Level", "font": {"color": TEXT_PRIMARY, "size": 12}},
            "font": {"color": TEXT_SECONDARY, "size": 11},
        },
    )
    return fig
