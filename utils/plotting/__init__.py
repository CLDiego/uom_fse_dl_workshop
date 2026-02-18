"""
Plotting utilities for the UoM Deep Learning Workshop.

This module provides various plotting functions and utilities for visualizing
machine learning concepts, data, and results, including:

- Loss and accuracy curve visualization
- Network architecture visualization
- Interactive visualizations for neural network concepts
- Distribution plots 
- Custom formatting and styling utilities
- Matplotlib font management and configuration

Key Components:
- plots: Core plotting functions for visualizing model performance and architectures
- formatting: Utilities for consistent and aesthetically pleasing plot styling
- fonts: Custom font loading and management
- interactive: Interactive widgets and visualizations for educational purposes
"""

__all__ = []

from .fonts import *
from .formatting import *
from .plots import *
from .interactive import *

from .fonts import __all__ as fonts_all
from .formatting import __all__ as formatting_all
from .plots import __all__ as plots_all
from .interactive import __all__ as interactive_all
from .se02_plots import __all__ as se02_plots_all

__all__.extend(fonts_all)
__all__.extend(formatting_all)
__all__.extend(plots_all)
__all__.extend(interactive_all)

from .plots import plot_loss, plot_distribution, visualize_network_nx
from .formatting import format_tick_label, format_axis_ticks, format_spines, make_fig_pretty
from .fonts import load_font
from .interactive import create_interactive_neuron_visualizer
from .se02_plots import plot_2d_classification, plot_training_loss, plot_model_comparison

__all__ = [
    "plot_loss", "plot_distribution", "visualize_network_nx",
    "format_tick_label", "format_axis_ticks", "format_spines", "make_fig_pretty",
    "load_font", "create_interactive_neuron_visualizer"
]

