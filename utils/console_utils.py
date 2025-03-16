"""
Console utilities for enhanced terminal output
"""

import sys
import os

# Create utils directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Try to import Rich console, install if necessary
try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
except ImportError:
    print("Installing Rich console library...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint

# Create console instance
console = Console()


def print_header(text):
    """Print a header with a styled panel"""
    console.print(Panel(text, style="bold blue"))


def print_success(text):
    """Print a success message in green"""
    console.print(f"✅ {text}", style="green")


def print_error(text):
    """Print an error message in red"""
    console.print(f"❌ {text}", style="bold red")


def print_warning(text):
    """Print a warning message in yellow"""
    console.print(f"⚠️ {text}", style="yellow")


def print_info(text):
    """Print an info message in cyan"""
    console.print(f"ℹ️ {text}", style="cyan")


def print_section(text):
    """Print a section header"""
    console.print(f"\n[bold magenta]== {text} ==[/bold magenta]")


def create_progress_bar(description="Processing"):
    """Create and return a progress bar"""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def display_table(title, data, columns):
    """Display data in a formatted table"""
    table = Table(title=title)

    # Add columns to table
    for column in columns:
        table.add_column(column, style="cyan")

    # Add rows to table
    for row in data:
        table.add_row(*[str(item) for item in row])

    console.print(table)
