# visualizer/user_dashboard/overall_dashboard.py
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class OverallDashboard:
    """
    Simple combined dashboard to present reliability, sentiment, and a small relevance heatmap.
    Expects a dict with keys: reliability, sentiment (counts), relevance_matrix.
    """

    def render(self, summary: dict):
        # Build a 2x2 layout
        fig = make_subplots(rows=2, cols=2, subplot_titles=[
            "Reliability", "Sentiment Distribution", "Relevance Heatmap", "Notes"
        ])

        # Reliability (gauge)
        reli = summary.get("reliability", 0.0)
        fig.add_trace(go.Indicator(mode="gauge+number", value=reli, gauge={'axis': {'range': [0,1]}}, title="Reliability"), row=1, col=1)

        # Sentiment bar
        sent = summary.get("sentiment", {})
        fig.add_trace(go.Bar(x=list(sent.keys()), y=list(sent.values()), name="Sentiment"), row=1, col=2)

        # Relevance heatmap
        rel = summary.get("relevance_matrix")
        if rel is not None:
            import numpy as np
            fig.add_trace(go.Heatmap(z=rel, colorscale="Viridis", showscale=True), row=2, col=1)

        # Notes
        notes = summary.get("notes", "")
        fig.add_trace(go.Scatter(x=[0], y=[0], text=[notes], mode="text"), row=2, col=2)

        fig.update_layout(height=800, width=1000, title_text="Interview Overview Dashboard")
        fig.show()
        return fig
