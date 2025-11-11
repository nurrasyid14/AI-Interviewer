# visualizer/user_dashboard/relevance_heatmap.py
import plotly.express as px
import pandas as pd
from typing import Sequence

class RelevanceHeatmap:
    """
    Visualize a matrix of relevance scores (questions x answers).
    Accepts a 2D list/array or pandas DataFrame.
    """

    def plot(self, matrix, x_labels=None, y_labels=None, show: bool = True):
        df = pd.DataFrame(matrix, index=y_labels, columns=x_labels)
        fig = px.imshow(df, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis")
        fig.update_layout(title="Relevance Heatmap", xaxis_title="Questions", yaxis_title="Answers")
        if show:
            fig.show()
        return fig
