"""Interactive Plotly chart components for the dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_risk_distribution(counts_dict: dict) -> go.Figure:
    """Plot risk category distribution donut chart."""
    labels = list(counts_dict.keys())
    values = list(counts_dict.values())
    colors = {
        "CRITICAL": "#ef4444",
        "HIGH": "#f59e0b",
        "MEDIUM": "#3b82f6",
        "LOW": "#10b981",
    }
    color_seq = [colors.get(risk_lvl, "#94a3b8") for risk_lvl in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker={"colors": color_seq, "line": {"color": "#0f172a", "width": 2}},
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
        showlegend=False,
    )
    return fig


def plot_churn_by_contract(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing churn rate across contract types."""
    if "Contract" not in df.columns or "Churn" not in df.columns:
        return go.Figure()

    churn_by_contract = df.groupby("Contract")["Churn"].mean().reset_index()
    churn_by_contract["ChurnRatePct"] = churn_by_contract["Churn"] * 100

    fig = px.bar(
        churn_by_contract,
        x="Contract",
        y="ChurnRatePct",
        text_auto=".1f",
        labels={"ChurnRatePct": "Churn Rate (%)", "Contract": "Contract Type"},
        color="ChurnRatePct",
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        coloraxis_showscale=False,
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
        margin={"t": 30, "b": 20, "l": 20, "r": 20},
    )
    return fig


def plot_churn_by_tenure_bucket(df: pd.DataFrame) -> go.Figure:
    """Bar chart of churn by tenure bucket."""
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
        labels={"ChurnRatePct": "Churn Rate (%)", "tenure_bucket": "Tenure Cohort"},
        color="ChurnRatePct",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        coloraxis_showscale=False,
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
        margin={"t": 30, "b": 20, "l": 20, "r": 20},
    )
    return fig


def plot_priority_scatter(df_results: pd.DataFrame) -> go.Figure:
    """Scatter plot: Churn Probability vs CLV sized by Retention Priority Score."""
    fig = px.scatter(
        df_results,
        x="churn_probability",
        y="clv",
        size="retention_priority_score",
        color="risk_level",
        hover_data=["customer_id", "retention_priority_score"],
        color_discrete_map={
            "CRITICAL": "#ef4444",
            "HIGH": "#f59e0b",
            "MEDIUM": "#3b82f6",
            "LOW": "#10b981",
        },
        labels={
            "churn_probability": "Churn Probability",
            "clv": "Customer Lifetime Value ($)",
            "risk_level": "Risk Tier",
        },
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        xaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
    )
    return fig
