"""Centralized Warm Light Enterprise Design Tokens.

Defines the exact color, typography, surface, border, and semantic risk tokens
used across the Telco Retention Intelligence platform.
"""

from typing import Dict

# Neutral & Surface Tokens
COLOR_PAGE_BG = "#F5F0E7"         # Warm Ivory
COLOR_PRIMARY_SURFACE = "#FFFDF8"  # Cream / Warm White
COLOR_SECONDARY_SURFACE = "#F0E9DD" # Warm Sand / Subtle Neutral
COLOR_BORDER = "#DED4C5"          # Warm Muted Border
COLOR_BORDER_LIGHT = "#EADBCE"    # Soft Inner Border

# Typography Tokens
COLOR_PRIMARY_TEXT = "#2D2924"    # Espresso / Dark Charcoal
COLOR_SECONDARY_TEXT = "#6F675D"  # Muted Warm Charcoal
COLOR_TERTIARY_TEXT = "#948A7D"   # Disabled / Tertiary Text

# Brand & Accent Tokens
COLOR_PRIMARY_BRAND = "#5E6B4A"   # Muted Olive / Sage
COLOR_PRIMARY_BRAND_DARK = "#465238" # Deep Olive (Hover / Active)
COLOR_SECONDARY_ACCENT = "#A56B4F" # Restrained Terracotta

# Semantic Risk Tokens
COLOR_RISK_LOW = "#56735A"        # Sage Green (Low Risk / Healthy)
COLOR_RISK_MEDIUM = "#B18445"     # Ochre / Amber (Medium Risk)
COLOR_RISK_HIGH = "#B96745"       # Terracotta (High Risk)
COLOR_RISK_CRITICAL = "#9F3F3F"   # Muted Deep Red (Critical Risk)

# Risk Badge Styling Specifications
RISK_BADGE_CONFIG: Dict[str, Dict[str, str]] = {
    "CRITICAL": {
        "bg": "#FAF0F0",
        "border": "#E8B8B8",
        "text": "#7A2B2B",
        "label": "CRITICAL RISK",
        "desc": "High probability of service cancellation. Immediate intervention suggested.",
    },
    "HIGH": {
        "bg": "#FBF2ED",
        "border": "#ECC3B0",
        "text": "#8A4123",
        "label": "HIGH RISK",
        "desc": "Elevated churn risk. Proactive retention review recommended.",
    },
    "MEDIUM": {
        "bg": "#FBF6EE",
        "border": "#E6D0AA",
        "text": "#7D5824",
        "label": "MEDIUM RISK",
        "desc": "Moderate churn probability. Monitor billing and usage changes.",
    },
    "LOW": {
        "bg": "#F1F5F0",
        "border": "#C6D8C5",
        "text": "#3D5440",
        "label": "LOW RISK",
        "desc": "Stable account profile. Routine customer care advised.",
    },
}

# Standard Typography Stack
FONT_FAMILY = 'Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
