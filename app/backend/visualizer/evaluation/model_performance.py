# visualizer/evaluation/model_performance.py
from typing import Dict
import plotly.graph_objects as go
import plotly.express as px

class ModelPerformanceVisualizer:
    """
    Plot training/validation curves (loss, accuracy) using Plotly.
    history: dict-like with keys like 'loss','val_loss','accuracy','val_accuracy'
    """

    def plot_training_curves(self, history: Dict[str, list], show: bool = True):
        fig = go.Figure()
        # Loss
        if "loss" in history:
            fig.add_trace(go.Scatter(y=history["loss"], mode="lines+markers", name="loss"))
        if "val_loss" in history:
            fig.add_trace(go.Scatter(y=history["val_loss"], mode="lines+markers", name="val_loss"))
        # Accuracy
        if "accuracy" in history:
            fig.add_trace(go.Scatter(y=history["accuracy"], mode="lines+markers", name="accuracy", yaxis="y2"))
        if "val_accuracy" in history:
            fig.add_trace(go.Scatter(y=history["val_accuracy"], mode="lines+markers", name="val_accuracy", yaxis="y2"))

        # layout with secondary y-axis
        fig.update_layout(
            title="Training Curves",
            xaxis_title="Epoch",
            yaxis=dict(title="Loss"),
            yaxis2=dict(title="Accuracy", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        if show:
            fig.show()
        return fig
