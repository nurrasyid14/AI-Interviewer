# visualizer/shared_components/plotly_theme.py
import plotly.io as pio

def apply_plotly_theme(template_name: str = "plotly_white"):
    """
    Apply a simple Plotly template globally.
    You can extend this to set fonts, colors, etc.
    """
    pio.templates.default = template_name
