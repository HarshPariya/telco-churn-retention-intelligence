"""Interactive Plotly chart components configured for warm light enterprise visual system."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.tokens import (
    COLOR_BORDER,
    COLOR_PRIMARY_BRAND,
    COLOR_PRIMARY_TEXT,
    COLOR_RISK_CRITICAL,
    COLOR_RISK_HIGH,
    COLOR_RISK_LOW,
    COLOR_RISK_MEDIUM,
    COLOR_SECONDARY_TEXT,
    FONT_FAMILY,
)

# Semantic Risk Mapping
RISK_COLORS = {
    "CRITICAL": COLOR_RISK_CRITICAL,
    "HIGH": COLOR_RISK_HIGH,
    "MEDIUM": COLOR_RISK_MEDIUM,
    "LOW": COLOR_RISK_LOW,
}


def plot_risk_distribution(counts_dict: dict) -> go.Figure:
    """Plot risk category distribution donut chart with warm enterprise styling."""
    labels = list(counts_dict.keys())
    values = list(counts_dict.values())
    color_seq = [RISK_COLORS.get(lvl, "#948A7D") for lvl in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker={"colors": color_seq, "line": {"color": "#FFFDF8", "width": 2}},
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                texttemplate="<b>%{label}</b><br>%{percent}",
                textfont={"color": "#FFFFFF", "size": 13, "family": FONT_FAMILY},
                insidetextorientation="horizontal",
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_PRIMARY_TEXT, "family": FONT_FAMILY},
        margin={"t": 16, "b": 16, "l": 16, "r": 16},
        showlegend=False,
    )
    return fig


def plot_churn_by_contract(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing observed churn rate across contract types with warm olive/terracotta emphasis."""
    if "Contract" not in df.columns or "Churn" not in df.columns:
        return go.Figure()

    churn_by_contract = df.groupby("Contract")["Churn"].mean().reset_index()
    churn_by_contract["ChurnRatePct"] = churn_by_contract["Churn"] * 100

    # Sort so Month-to-month comes first
    contract_order = ["Month-to-month", "One year", "Two year"]
    churn_by_contract["sort_order"] = churn_by_contract["Contract"].apply(
        lambda x: contract_order.index(x) if x in contract_order else 99
    )
    churn_by_contract = churn_by_contract.sort_values("sort_order")

    colors = [COLOR_RISK_HIGH, COLOR_PRIMARY_BRAND, COLOR_RISK_LOW]

    fig = px.bar(
        churn_by_contract,
        x="Contract",
        y="ChurnRatePct",
        text_auto=".1f",
        labels={"ChurnRatePct": "Observed Churn (%)", "Contract": "Contract Type"},
    )
    fig.update_traces(
        marker_color=colors[: len(churn_by_contract)],
        marker_line_color=COLOR_BORDER,
        marker_line_width=1,
        textfont={"color": "#FFFDF8", "size": 12, "family": FONT_FAMILY},
        textposition="inside",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_PRIMARY_TEXT, "family": FONT_FAMILY},
        xaxis={
            "showgrid": False,
            "title": {"font": {"color": COLOR_SECONDARY_TEXT, "size": 12, "family": FONT_FAMILY}},
            "tickfont": {"color": COLOR_PRIMARY_TEXT, "size": 12, "family": FONT_FAMILY},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": COLOR_BORDER,
            "title": {"font": {"color": COLOR_SECONDARY_TEXT, "size": 12, "family": FONT_FAMILY}},
            "tickfont": {"color": COLOR_SECONDARY_TEXT, "size": 11, "family": FONT_FAMILY},
        },
        margin={"t": 20, "b": 16, "l": 16, "r": 16},
    )
    return fig


def plot_churn_by_tenure_bucket(df: pd.DataFrame) -> go.Figure:
    """Bar chart of observed churn across tenure cohorts using a restrained warm tonal progression."""
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
    )
    fig.update_traces(
        marker_color=COLOR_PRIMARY_BRAND,
        marker_line_color=COLOR_BORDER,
        marker_line_width=1,
        textfont={"color": "#FFFDF8", "size": 12, "family": FONT_FAMILY},
        textposition="inside",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_PRIMARY_TEXT, "family": FONT_FAMILY},
        xaxis={
            "showgrid": False,
            "title": {"font": {"color": COLOR_SECONDARY_TEXT, "size": 12, "family": FONT_FAMILY}},
            "tickfont": {"color": COLOR_PRIMARY_TEXT, "size": 12, "family": FONT_FAMILY},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": COLOR_BORDER,
            "title": {"font": {"color": COLOR_SECONDARY_TEXT, "size": 12, "family": FONT_FAMILY}},
            "tickfont": {"color": COLOR_SECONDARY_TEXT, "size": 11, "family": FONT_FAMILY},
        },
        margin={"t": 20, "b": 16, "l": 16, "r": 16},
    )
    return fig


def plot_priority_scatter(df_results: pd.DataFrame) -> go.Figure:
    """Scatter plot: Churn Probability vs Customer Lifetime Value sized by Priority Score in warm palette."""
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
    fig.update_traces(
        marker={
            "opacity": 0.85,
            "line": {"width": 1, "color": "rgba(45, 41, 36, 0.35)"},
        }
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_PRIMARY_TEXT, "family": FONT_FAMILY},
        xaxis={
            "showgrid": True,
            "gridcolor": COLOR_BORDER,
            "title": {"font": {"color": COLOR_SECONDARY_TEXT, "size": 12, "family": FONT_FAMILY}},
            "tickfont": {"color": COLOR_SECONDARY_TEXT, "size": 11, "family": FONT_FAMILY},
            "tickformat": ".0%",
        },
        yaxis={
            "showgrid": True,
            "gridcolor": COLOR_BORDER,
            "title": {"font": {"color": COLOR_SECONDARY_TEXT, "size": 12, "family": FONT_FAMILY}},
            "tickfont": {"color": COLOR_SECONDARY_TEXT, "size": 11, "family": FONT_FAMILY},
            "tickprefix": "$",
        },
        margin={"t": 16, "b": 16, "l": 16, "r": 16},
        legend={
            "title": {"text": "Risk Level", "font": {"color": COLOR_PRIMARY_TEXT, "size": 12, "family": FONT_FAMILY}},
            "font": {"color": COLOR_SECONDARY_TEXT, "size": 11, "family": FONT_FAMILY},
        },
    )
    return fig
