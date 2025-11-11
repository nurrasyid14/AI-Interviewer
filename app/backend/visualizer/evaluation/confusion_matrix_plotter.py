# visualizer/evaluation/confusion_matrix_plotter.py
from typing import List, Optional
import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

class ConfusionMatrixPlotter:
    """
    Confusion matrix visualization using Plotly heatmap via figure_factory.
    """

    def plot(self, y_true, y_pred, labels: Optional[List[str]] = None, show: bool = True):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if labels is None:
            # infer labels from unique y_true/y_pred
            labels = list(np.unique(np.concatenate([np.array(y_true), np.array(y_pred)])))
        # normalize percent matrix
        with_perc = (cm.astype("float") / (cm.sum(axis=1)[:, None] + 1e-9)) * 100
        txt = [[f"{cm[i][j]} ({with_perc[i][j]:.1f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
        fig = ff.create_annotated_heatmap(
            z=cm.tolist(),
            x=labels,
            y=labels,
            annotation_text=txt,
            colorscale="Blues",
            showscale=True
        )
        fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        if show:
            fig.show()
        return fig
