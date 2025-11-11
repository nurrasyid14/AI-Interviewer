# visualizer/visualizer_pipeline.py
from typing import Optional, Any, Dict
from .evaluation.model_performance import ModelPerformanceVisualizer
from .evaluation.confusion_matrix_plotter import ConfusionMatrixPlotter
from .evaluation.reliability_distribution import ReliabilityVisualizer
from .user_dashboard.overall_dashboard import OverallDashboard
from .shared_components.plotly_theme import apply_plotly_theme

class VisualizerPipeline:
    """
    Central orchestrator for visualizations.
    mode: "evaluation" | "dashboard"
    """

    def __init__(self, mode: str = "evaluation", theme: str = "simple_white"):
        self.mode = mode
        apply_plotly_theme(theme)
        # Evaluation visuals
        self.model_viz = ModelPerformanceVisualizer()
        self.cm_viz = ConfusionMatrixPlotter()
        self.reliability_viz = ReliabilityVisualizer()
        # Dashboard visuals
        self.dashboard = OverallDashboard()

    # Evaluation endpoints
    def show_model_performance(self, history: Dict[str, list], show: bool = True):
        return self.model_viz.plot_training_curves(history, show=show)

    def show_confusion_matrix(self, y_true, y_pred, labels=None, show: bool = True):
        return self.cm_viz.plot(y_true, y_pred, labels=labels, show=show)

    def show_reliability_distribution(self, probabilities, show: bool = True):
        return self.reliability_viz.plot_confidence_distribution(probabilities, show=show)

    # Dashboard endpoints
    def generate_overall_dashboard(self, interview_summary: Dict[str, Any], output_html: Optional[str]=None):
        fig = self.dashboard.render(interview_summary)
        if output_html:
            fig.write_html(output_html)
        return fig
