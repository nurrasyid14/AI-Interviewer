# visualizer/evaluation/reliability_distribution.py
from typing import Sequence
import plotly.express as px
import pandas as pd

class ReliabilityVisualizer:
    """
    Visualizes predicted probabilities / reliability scores distribution and calibration-like views.
    """

    def plot_confidence_distribution(self, probabilities: Sequence[float], bins: int = 20, show: bool = True):
        df = pd.DataFrame({"probability": probabilities})
        fig = px.histogram(df, x="probability", nbins=bins, title="Prediction Confidence Distribution")
        fig.update_layout(xaxis_title="Probability / Reliability", yaxis_title="Count")
        if show:
            fig.show()
        return fig

    def plot_reliability_vs_outcome(self, probabilities: Sequence[float], outcomes: Sequence[int], show: bool = True):
        # outcomes assumed 0/1
        df = pd.DataFrame({"probability": probabilities, "outcome": outcomes})
        fig = px.box(df, x="outcome", y="probability", title="Reliability by Outcome (0/1)")
        fig.update_layout(xaxis_title="Outcome", yaxis_title="Predicted Probability")
        if show:
            fig.show()
        return fig
