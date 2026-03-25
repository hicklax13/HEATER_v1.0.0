"""
UI Analytics Badge — renders AnalyticsContext transparency into Streamlit.

Replaces the current pattern of showing recommendations with no indication
of what produced them. Every recommendation now carries a visible badge
showing data quality, module execution, and honest confidence.

Usage in any page:
    from src.analytics_context import AnalyticsContext
    from src.ui_analytics_badge import render_analytics_badge

    ctx = AnalyticsContext(pipeline="trade_engine")
    # ... pipeline runs, stamps ctx ...
    render_analytics_badge(ctx)  # Renders badge in sidebar or inline
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analytics_context import AnalyticsContext, ConfidenceTier, DataQuality, ModuleStatus
from src.ui_shared import THEME as T

if TYPE_CHECKING:
    pass


def build_analytics_badge_html(ctx: AnalyticsContext) -> str:
    """
    Build HTML for the analytics transparency badge.

    Shows:
    - Confidence tier (color-coded)
    - Quality score (0-100)
    - Module execution summary
    - Data freshness warnings
    - Expandable detail panel
    """
    tier = ctx.confidence_tier
    score = ctx.quality_score
    warnings = ctx.user_warnings
    summary = ctx.modules_summary

    # Tier colors
    tier_colors = {
        ConfidenceTier.HIGH: T.get("cool", "#457b9d"),
        ConfidenceTier.MEDIUM: T.get("hot", "#ff6d00"),
        ConfidenceTier.LOW: T.get("primary", "#e63946"),
        ConfidenceTier.EXPERIMENTAL: "#9c27b0",
    }
    tier_labels = {
        ConfidenceTier.HIGH: "High Input Quality",
        ConfidenceTier.MEDIUM: "Moderate Input Quality",
        ConfidenceTier.LOW: "Low Input Quality",
        ConfidenceTier.EXPERIMENTAL: "Experimental",
    }

    color = tier_colors.get(tier, T.get("tx", "#2b2d42"))
    label = tier_labels.get(tier, "Unknown")
    score_pct = int(score * 100)

    # Module breakdown
    module_html = ""
    if ctx.modules:
        module_rows = []
        for name, mod in ctx.modules.items():
            status_icon = {
                ModuleStatus.EXECUTED: f'<span style="color:{T.get("cool", "#457b9d")}">ran</span>',
                ModuleStatus.FALLBACK: f'<span style="color:{T.get("hot", "#ff6d00")}">fallback</span>',
                ModuleStatus.SKIPPED: '<span style="color:#999">skipped</span>',
                ModuleStatus.DISABLED: f'<span style="color:{T.get("primary", "#e63946")}">disabled</span>',
                ModuleStatus.ERROR: f'<span style="color:{T.get("primary", "#e63946")}">error</span>',
                ModuleStatus.NOT_APPLICABLE: '<span style="color:#999">n/a</span>',
            }.get(mod.status, "?")

            reason = ""
            if mod.fallback_reason:
                reason = f' <span style="color:#999;font-size:9px">({mod.fallback_reason})</span>'

            module_rows.append(
                f'<tr><td style="font-size:10px;padding:1px 4px">{name}</td>'
                f'<td style="font-size:10px;padding:1px 4px">{status_icon}{reason}</td>'
                f'<td style="font-size:10px;padding:1px 4px;text-align:right">'
                f"{mod.execution_ms:.0f}ms</td></tr>"
            )

        module_html = (
            '<table style="width:100%;border-collapse:collapse;margin-top:4px">' + "".join(module_rows) + "</table>"
        )

    # Data source breakdown
    data_html = ""
    if ctx.data_sources:
        data_rows = []
        for name, ds in ctx.data_sources.items():
            quality_badge = {
                DataQuality.LIVE: f'<span style="color:{T.get("cool", "#457b9d")}">live</span>',
                DataQuality.STALE: f'<span style="color:{T.get("hot", "#ff6d00")}">stale</span>',
                DataQuality.SAMPLE: f'<span style="color:{T.get("primary", "#e63946")}">sample</span>',
                DataQuality.MISSING: f'<span style="color:{T.get("primary", "#e63946")}">missing</span>',
                DataQuality.HARDCODED: '<span style="color:#999">hardcoded</span>',
            }.get(ds.quality, "?")

            age = ""
            if ds.age_hours is not None:
                if ds.age_hours < 1:
                    age = f"{ds.age_hours * 60:.0f}m ago"
                elif ds.age_hours < 24:
                    age = f"{ds.age_hours:.0f}h ago"
                else:
                    age = f"{ds.age_hours / 24:.0f}d ago"

            data_rows.append(
                f'<tr><td style="font-size:10px;padding:1px 4px">{name}</td>'
                f'<td style="font-size:10px;padding:1px 4px">{quality_badge}</td>'
                f'<td style="font-size:10px;padding:1px 4px;text-align:right">{age}</td></tr>'
            )

        data_html = (
            '<table style="width:100%;border-collapse:collapse;margin-top:4px">' + "".join(data_rows) + "</table>"
        )

    # Warnings
    warning_html = ""
    if warnings:
        warning_items = "".join(
            f'<li style="font-size:10px;color:{T.get("primary", "#e63946")};margin:1px 0">{w}</li>'
            for w in warnings[:5]  # Cap at 5 warnings
        )
        warning_html = f'<ul style="margin:4px 0;padding-left:16px">{warning_items}</ul>'

    # Assemble badge
    html = f"""
    <div style="
        border: 1px solid {color}40;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 8px 0;
        background: {color}08;
        font-family: monospace;
    ">
        <div style="display:flex;align-items:center;gap:8px">
            <div style="
                width:8px;height:8px;border-radius:50%;
                background:{color};flex-shrink:0;
            "></div>
            <span style="font-size:11px;font-weight:600;color:{color}">
                {label}
            </span>
            <span style="font-size:10px;color:#999;margin-left:auto">
                {summary}
            </span>
        </div>
        {warning_html}
        {module_html}
        {data_html}
    </div>
    """
    return html


def render_analytics_badge(ctx: AnalyticsContext) -> None:
    """
    Render the analytics transparency badge in Streamlit.

    Call this after any recommendation is displayed.
    Shows a collapsible panel with full pipeline transparency.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    tier = ctx.confidence_tier
    tier_emoji_map = {
        ConfidenceTier.HIGH: "Analysis Quality: High",
        ConfidenceTier.MEDIUM: "Analysis Quality: Moderate",
        ConfidenceTier.LOW: "Analysis Quality: Low",
        ConfidenceTier.EXPERIMENTAL: "Analysis Quality: Experimental",
    }
    expander_label = tier_emoji_map.get(tier, "Analysis Quality")

    with st.expander(expander_label, expanded=False):
        html = build_analytics_badge_html(ctx)
        st.markdown(html, unsafe_allow_html=True)

        # Show quality score as a progress bar
        score = ctx.quality_score
        st.progress(score, text=f"Input quality: {int(score * 100)}%")
