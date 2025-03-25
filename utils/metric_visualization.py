import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import interpolate
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MetricsVisualizer:
    """
    Class for visualizing metrics from YOLO ensemble detection models.
    Supports visualizing precision, recall, mAP@0.5, and mAP@0.5-0.95.
    """

    def __init__(self, output_dir="metrics"):
        """
        Initialize the metrics visualizer.

        Args:
            output_dir (str): Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set default style for visualizations
        sns.set_style("whitegrid")
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.figsize": (10, 6),
            }
        )

        # Color palette for consistent visualization
        self.color_palette = sns.color_palette("muted", 10)

    def generate_model_comparison_chart(
        self, metrics_data, metric_name, title=None, save_path=None, interactive=False
    ):
        """
        Generate a bar chart comparing a specific metric across models.

        Args:
            metrics_data (dict): Dictionary with model names as keys and metrics as values
            metric_name (str): The name of the metric to plot
            title (str, optional): Custom title for the chart
            save_path (str, optional): Path to save the visualization
            interactive (bool): Whether to create an interactive plotly visualization

        Returns:
            fig: The generated figure object
        """
        models = list(metrics_data.keys())
        metric_values = [metrics_data[model].get(metric_name, 0) for model in models]

        if interactive:
            # Create interactive plotly chart
            fig = px.bar(
                x=models,
                y=metric_values,
                labels={"x": "Model", "y": metric_name},
                title=title or f"{metric_name} Comparison Across Models",
                color=metric_values,
                color_continuous_scale="Viridis",
            )

            if save_path:
                fig.write_html(f"{save_path}.html")
                fig.write_image(f"{save_path}.png")

            return fig
        else:
            # Create static matplotlib chart
            fig, ax = plt.subplots()
            bars = ax.bar(models, metric_values, color=self.color_palette)

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )

            ax.set_ylabel(metric_name)
            ax.set_title(title or f"{metric_name} Comparison Across Models")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            if save_path:
                plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

            return fig

    def generate_precision_recall_curve(
        self, pr_data, title=None, save_path=None, interactive=False
    ):
        """
        Generate precision-recall curves for each model.

        Args:
            pr_data (dict): Dictionary with model names as keys and precision-recall data as values
                Each model should have 'precision' and 'recall' arrays
            title (str, optional): Custom title for the chart
            save_path (str, optional): Path to save the visualization
            interactive (bool): Whether to create an interactive plotly visualization

        Returns:
            fig: The generated figure object
        """
        if interactive:
            fig = go.Figure()

            for i, (model_name, data) in enumerate(pr_data.items()):
                fig.add_trace(
                    go.Scatter(
                        x=data["recall"],
                        y=data["precision"],
                        mode="lines",
                        name=model_name,
                        fill="tozeroy",
                        opacity=0.5,
                    )
                )

            fig.update_layout(
                title=title or "Precision-Recall Curves",
                xaxis_title="Recall",
                yaxis_title="Precision",
                legend_title="Models",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
            )

            if save_path:
                fig.write_html(f"{save_path}.html")
                fig.write_image(f"{save_path}.png")

            return fig

        else:
            fig, ax = plt.subplots()

            for i, (model_name, data) in enumerate(pr_data.items()):
                ax.plot(
                    data["recall"],
                    data["precision"],
                    label=model_name,
                    color=self.color_palette[i],
                )
                ax.fill_between(
                    data["recall"],
                    data["precision"],
                    alpha=0.1,
                    color=self.color_palette[i],
                )

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(title or "Precision-Recall Curves")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.legend()

            if save_path:
                plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

            return fig

    def generate_metrics_table(self, metrics_data, save_path=None):
        """
        Generate a summary table of metrics.

        Args:
            metrics_data (dict): Dictionary with model names as keys and metrics as values
            save_path (str, optional): Path to save the table

        Returns:
            DataFrame: Pandas DataFrame containing the metrics table
        """
        # Extract common metrics
        table_data = {}

        for model, metrics in metrics_data.items():
            table_data[model] = {
                "mAP@0.5": metrics.get("mAP50", 0),
                "mAP@0.5-0.95": metrics.get("mAP50-95", 0),
                "Precision": metrics.get("precision", 0),
                "Recall": metrics.get("recall", 0),
                "F1-Score": metrics.get("f1", 0),
                "Inference Time (ms)": metrics.get("inference_time", 0),
            }

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(table_data, orient="index")

        # Save table if path is provided
        if save_path:
            # Save as CSV
            df.to_csv(f"{save_path}.csv")

            # Save as styled HTML
            styled_df = df.style.background_gradient(cmap="viridis")
            styled_df.to_html(f"{save_path}.html")

        return df

    def save_metrics_data(self, metrics_data, save_path):
        """
        Save metrics data to a JSON file.

        Args:
            metrics_data (dict): Dictionary with model names as keys and metrics as values
            save_path (str): Path to save the JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}

        for model, metrics in metrics_data.items():
            serializable_data[model] = {}
            for metric_name, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_data[model][metric_name] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    serializable_data[model][metric_name] = float(value)
                else:
                    serializable_data[model][metric_name] = value

        # Include timestamp
        serializable_data["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }

        with open(save_path, "w") as f:
            json.dump(serializable_data, f, indent=4)

    def load_metrics_data(self, load_path):
        """
        Load metrics data from a JSON file.

        Args:
            load_path (str): Path to the JSON file

        Returns:
            dict: Dictionary with model names as keys and metrics as values
        """
        with open(load_path, "r") as f:
            data = json.load(f)

        # Remove metadata if present
        if "metadata" in data:
            data.pop("metadata")

        return data

    def generate_metrics_report(self, metrics_data, output_dir=None):
        """
        Generate a comprehensive metrics report with multiple visualizations.

        Args:
            metrics_data (dict): Dictionary with model names as keys and metrics as values
            output_dir (str, optional): Directory to save the report outputs

        Returns:
            dict: Dictionary with paths to generated visualizations
        """
        if output_dir is None:
            output_dir = self.output_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(output_dir, f"metrics_report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)

        report_files = {}

        # Generate and save individual metric comparisons
        for metric in ["mAP50", "mAP50-95", "precision", "recall", "f1"]:
            if all(metric in metrics for metrics in metrics_data.values()):
                metric_path = os.path.join(report_dir, f"{metric}_comparison")
                self.generate_model_comparison_chart(
                    metrics_data, metric, save_path=metric_path, interactive=True
                )
                report_files[f"{metric}_chart"] = metric_path + ".html"

        # Generate precision-recall curves if data is available
        pr_data = {}
        for model, metrics in metrics_data.items():
            if "precision_curve" in metrics and "recall_curve" in metrics:
                pr_data[model] = {
                    "precision": metrics["precision_curve"],
                    "recall": metrics["recall_curve"],
                }

        if pr_data:
            pr_path = os.path.join(report_dir, "precision_recall_curves")
            self.generate_precision_recall_curve(
                pr_data, save_path=pr_path, interactive=True
            )
            report_files["pr_curves"] = pr_path + ".html"

        # Generate metrics table
        table_path = os.path.join(report_dir, "metrics_summary")
        self.generate_metrics_table(metrics_data, save_path=table_path)
        report_files["metrics_table"] = table_path + ".html"

        # Save the raw metrics data
        data_path = os.path.join(report_dir, "metrics_data.json")
        self.save_metrics_data(metrics_data, data_path)
        report_files["metrics_data"] = data_path

        return report_files
