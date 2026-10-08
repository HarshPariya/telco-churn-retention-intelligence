"""Compatibility wrapper re-exporting cards from dashboard.components.cards."""

from dashboard.components.cards import (
    render_driver_card,
    render_empty_state,
    render_metric_card,
    render_risk_badge,
)

__all__ = [
    "render_driver_card",
    "render_empty_state",
    "render_metric_card",
    "render_risk_badge",
]
