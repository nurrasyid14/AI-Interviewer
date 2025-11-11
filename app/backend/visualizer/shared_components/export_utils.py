# visualizer/shared_components/export_utils.py
from typing import Optional
import os

class ExportUtils:
    @staticmethod
    def save_html(fig, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.write_html(path)

    @staticmethod
    def save_png(fig, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # requires kaleido installed (`pip install kaleido`)
        fig.write_image(path)
    @staticmethod
    def export_figure(fig, path: str, format: Optional[str] = "html"):
        """
        Export a Plotly figure to the specified path in the given format.
        Supported formats: "html", "png"
        """
        if format == "html":
            ExportUtils.save_html(fig, path)
        elif format == "png":
            ExportUtils.save_png(fig, path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        