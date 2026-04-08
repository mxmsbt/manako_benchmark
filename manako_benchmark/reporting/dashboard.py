"""HTML report & dashboard generation."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Template

from ..evaluation.runner import BenchmarkResults
from ..evaluation.temporal import TemporalTracker


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manako Vision AI Benchmark Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 2rem; }
        .header { text-align: center; margin-bottom: 3rem; }
        .header h1 { font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #00d4ff, #7b2ff7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header .subtitle { color: #888; margin-top: 0.5rem; font-size: 1.1rem; }
        .header .timestamp { color: #555; font-size: 0.85rem; margin-top: 0.3rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .card { background: #141414; border: 1px solid #222; border-radius: 12px; padding: 1.5rem; }
        .card h2 { font-size: 1.1rem; color: #aaa; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }
        .metric-big { font-size: 3rem; font-weight: 700; }
        .metric-label { font-size: 0.9rem; color: #666; }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 0.75rem; border-bottom: 2px solid #333; color: #888; font-size: 0.85rem; text-transform: uppercase; }
        td { padding: 0.75rem; border-bottom: 1px solid #1a1a1a; }
        tr:hover { background: #1a1a1a; }
        .rank-1 { color: #00d4ff; font-weight: 700; }
        .rank-2 { color: #7b2ff7; font-weight: 600; }
        .rank-3 { color: #888; }
        .bar { height: 8px; border-radius: 4px; background: #222; position: relative; margin-top: 4px; }
        .bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #00d4ff, #7b2ff7); }
        .chart-container { width: 100%; margin: 1rem 0; }
        .section { margin-bottom: 2rem; }
        .section-title { font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #222; }
        .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
        .tag-sn44 { background: #00d4ff22; color: #00d4ff; }
        .tag-sam3 { background: #ff6b6b22; color: #ff6b6b; }
        .tag-roboflow { background: #7b2ff722; color: #7b2ff7; }
        .footer { text-align: center; color: #444; font-size: 0.8rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #1a1a1a; }
        .improvement { color: #00ff88; }
        .ceiling-gap { color: #ff6b6b; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Manako Vision AI Benchmark</h1>
        <div class="subtitle">Object Detection: SN44 vs SAM3 vs Roboflow | mAP@50</div>
        <div class="timestamp">Generated: {{ timestamp }}</div>
    </div>

    <!-- Summary Cards -->
    <div class="grid">
        {% for mr in model_results %}
        <div class="card">
            <h2><span class="tag tag-{{ mr.model_name }}">{{ mr.model_name | upper }}</span></h2>
            <div class="metric-big {% if loop.index == 1 %}rank-1{% elif loop.index == 2 %}rank-2{% else %}rank-3{% endif %}">
                {{ "%.1f" | format(mr.mAP50 * 100) }}%
            </div>
            <div class="metric-label">mAP@50</div>
            <div class="bar"><div class="bar-fill" style="width: {{ (mr.mAP50 * 100) | round(1) }}%"></div></div>
            <div style="margin-top: 0.75rem; font-size: 0.85rem; color: #666;">
                {{ mr.num_predictions }} detections | {{ "%.0f" | format(mr.avg_inference_ms) }}ms avg
            </div>
        </div>
        {% endfor %}
    </div>

    <!-- Comparison Chart -->
    <div class="section">
        <div class="section-title">Model Comparison</div>
        <div class="card">
            <div class="chart-container">{{ comparison_chart }}</div>
        </div>
    </div>

    <!-- Per-Class Breakdown -->
    <div class="section">
        <div class="section-title">Per-Class AP@50</div>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Class</th>
                        {% for mr in model_results %}
                        <th>{{ mr.model_name | upper }}</th>
                        {% endfor %}
                        <th>Gap to Ceiling</th>
                    </tr>
                </thead>
                <tbody>
                    {% for cls_id, cls_name in class_names.items() %}
                    <tr>
                        <td><strong>{{ cls_name }}</strong></td>
                        {% for mr in model_results %}
                        <td>{{ "%.1f" | format((mr.per_class_ap.get(cls_id, 0)) * 100) }}%</td>
                        {% endfor %}
                        <td>
                            {% if ceiling_gaps[cls_id] is defined %}
                            <span class="ceiling-gap">{{ "%.1f" | format(ceiling_gaps[cls_id] * 100) }}%</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <!-- Per-Class Chart -->
    <div class="section">
        <div class="card">
            <div class="chart-container">{{ per_class_chart }}</div>
        </div>
    </div>

    {% if temporal_chart %}
    <!-- Temporal Improvement -->
    <div class="section">
        <div class="section-title">Improvement Over Time (Subnet Training)</div>
        <div class="card">
            <div class="chart-container">{{ temporal_chart }}</div>
        </div>
        {% if improvement_summary %}
        <div class="card" style="margin-top: 1rem;">
            <h2>SN44 Training Progress</h2>
            <table>
                <tr><td>Initial mAP@50</td><td><strong>{{ "%.1f" | format(improvement_summary.initial_mAP50 * 100) }}%</strong></td></tr>
                <tr><td>Final mAP@50</td><td><strong>{{ "%.1f" | format(improvement_summary.final_mAP50 * 100) }}%</strong></td></tr>
                <tr><td>Absolute Improvement</td><td><span class="improvement">+{{ "%.1f" | format(improvement_summary.absolute_improvement * 100) }}%</span></td></tr>
                <tr><td>Relative Improvement</td><td><span class="improvement">+{{ "%.1f" | format(improvement_summary.relative_improvement_pct) }}%</span></td></tr>
                <tr><td>Checkpoints Evaluated</td><td>{{ improvement_summary.num_checkpoints }}</td></tr>
            </table>
        </div>
        {% endif %}
    </div>
    {% endif %}

    <!-- Dataset Info -->
    <div class="section">
        <div class="section-title">Dataset Information</div>
        <div class="card">
            <table>
                <tr><td>Images</td><td><strong>{{ dataset_info.num_images }}</strong></td></tr>
                <tr><td>Annotations</td><td><strong>{{ dataset_info.num_annotations }}</strong></td></tr>
                <tr><td>Categories</td><td><strong>{{ dataset_info.num_categories }}</strong></td></tr>
            </table>
        </div>
    </div>

    <!-- Detailed Results Table -->
    <div class="section">
        <div class="section-title">Full Results</div>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>mAP@50</th>
                        <th>Predictions</th>
                        <th>Avg Inference (ms)</th>
                        <th>Total Time (s)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for mr in model_results %}
                    <tr>
                        <td class="rank-{{ loop.index }}">#{{ loop.index }}</td>
                        <td><span class="tag tag-{{ mr.model_name }}">{{ mr.model_name | upper }}</span></td>
                        <td><strong>{{ "%.4f" | format(mr.mAP50) }}</strong></td>
                        <td>{{ mr.num_predictions }}</td>
                        <td>{{ "%.1f" | format(mr.avg_inference_ms) }}</td>
                        <td>{{ "%.2f" | format(mr.total_time_s) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <div class="footer">
        Manako Benchmark v0.1.0 | SN44 Bittensor Subnet | Web3-trained Vision AI
    </div>
</body>
</html>"""


MODEL_COLORS = {
    "sn44": "#00d4ff",
    "sam3": "#ff6b6b",
    "roboflow": "#7b2ff7",
}


def _make_comparison_chart(results: BenchmarkResults) -> str:
    """Bar chart comparing overall mAP@50 across models."""
    fig = go.Figure()
    for mr in results.model_results:
        color = MODEL_COLORS.get(mr.model_name, "#888")
        fig.add_trace(go.Bar(
            name=mr.model_name.upper(),
            x=[mr.model_name.upper()],
            y=[mr.mAP50 * 100],
            marker_color=color,
            text=[f"{mr.mAP50 * 100:.1f}%"],
            textposition="outside",
        ))
    fig.update_layout(
        title="Overall mAP@50 Comparison",
        yaxis_title="mAP@50 (%)",
        yaxis_range=[0, 105],
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _make_per_class_chart(results: BenchmarkResults, class_names: dict[int, str]) -> str:
    """Grouped bar chart: per-class AP@50 for each model."""
    fig = go.Figure()
    categories = list(class_names.values())
    for mr in results.model_results:
        color = MODEL_COLORS.get(mr.model_name, "#888")
        values = [mr.per_class_ap.get(cid, 0) * 100 for cid in class_names]
        fig.add_trace(go.Bar(
            name=mr.model_name.upper(),
            x=categories,
            y=values,
            marker_color=color,
        ))
    fig.update_layout(
        title="Per-Class AP@50",
        yaxis_title="AP@50 (%)",
        yaxis_range=[0, 105],
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _make_temporal_chart(tracker: TemporalTracker) -> str | None:
    """Line chart showing mAP@50 over checkpoints for each model."""
    if len(tracker.history) < 2:
        return None

    fig = go.Figure()
    model_names = set()
    for entry in tracker.history:
        model_names.update(entry["models"].keys())

    for model_name in sorted(model_names):
        timeline = tracker.get_model_timeline(model_name)
        if not timeline:
            continue
        color = MODEL_COLORS.get(model_name, "#888")
        x_labels = [t.get("checkpoint_tag") or t["timestamp"][:10] for t in timeline]
        y_values = [t["mAP50"] * 100 for t in timeline]
        fig.add_trace(go.Scatter(
            name=model_name.upper(),
            x=x_labels,
            y=y_values,
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=8),
        ))

    fig.update_layout(
        title="mAP@50 Improvement Over Subnet Training",
        xaxis_title="Checkpoint",
        yaxis_title="mAP@50 (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_report(
    results: BenchmarkResults,
    output_path: str | Path,
    tracker: TemporalTracker | None = None,
) -> Path:
    """Generate a full HTML benchmark report.

    Args:
        results: The benchmark results to report.
        output_path: Where to save the HTML file.
        tracker: Optional temporal tracker for improvement-over-time charts.

    Returns:
        Path to the generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort results by mAP@50 descending
    results.model_results.sort(key=lambda mr: mr.mAP50, reverse=True)

    # Collect class names
    class_names = {}
    if results.dataset_info.get("categories"):
        cats = results.dataset_info["categories"]
        if isinstance(cats, dict):
            class_names = {int(k): v for k, v in cats.items()}
        elif isinstance(cats, list):
            class_names = {c["id"]: c["name"] for c in cats}

    # Compute ceiling gaps (gap between SN44 and best model per class)
    ceiling_gaps = {}
    sn44_result = next((mr for mr in results.model_results if mr.model_name == "sn44"), None)
    best_result = results.model_results[0] if results.model_results else None
    if sn44_result and best_result and best_result.model_name != "sn44":
        for cls_id in class_names:
            ceiling = best_result.per_class_ap.get(cls_id, 0)
            sn44_val = sn44_result.per_class_ap.get(cls_id, 0)
            ceiling_gaps[cls_id] = ceiling - sn44_val

    # Build charts
    comparison_chart = _make_comparison_chart(results)
    per_class_chart = _make_per_class_chart(results, class_names)
    temporal_chart = _make_temporal_chart(tracker) if tracker else None
    improvement_summary = tracker.get_improvement_summary("sn44") if tracker else None

    # Render template
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        timestamp=results.timestamp,
        model_results=results.model_results,
        dataset_info=results.dataset_info,
        class_names=class_names,
        ceiling_gaps=ceiling_gaps,
        comparison_chart=comparison_chart,
        per_class_chart=per_class_chart,
        temporal_chart=temporal_chart,
        improvement_summary=improvement_summary,
    )

    output_path.write_text(html)
    print(f"Report saved to: {output_path}")
    return output_path
