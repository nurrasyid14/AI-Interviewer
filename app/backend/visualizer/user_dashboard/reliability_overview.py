# visualizer/user_dashboard/reliability_overview.py
import plotly.graph_objects as go

class ReliabilityOverview:
    """
    Simple card-style reliability overview. Expects a dict:
      {"reliability": float, "threshold": float, "details": {...}}
    """

    def render(self, data: dict, show: bool = True):
        reli = data.get("reliability", 0.0)
        threshold = data.get("threshold", 0.75)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=float(reli),
            delta={'reference': float(threshold)},
            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "darkblue"}},
            title={'text': "Reliability Score"}
        ))
        if show:
            gauge.show()
        return gauge
