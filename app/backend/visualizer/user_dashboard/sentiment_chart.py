# visualizer/user_dashboard/sentiment_chart.py
import plotly.express as px
import pandas as pd
from typing import Sequence

class SentimentChart:
    """
    Plot sentiment distribution or time-series.
    """

    def distribution(self, sentiments: Sequence[str], show: bool = True):
        df = pd.DataFrame({"sentiment": sentiments})
        fig = px.histogram(df, x="sentiment", title="Sentiment Distribution")
        if show:
            fig.show()
        return fig

    def timeseries(self, timestamps, scores, show: bool = True):
        df = pd.DataFrame({"ts": timestamps, "score": scores})
        fig = px.line(df, x="ts", y="score", title="Sentiment Over Time")
        if show:
            fig.show()
        return fig
