"""
SE02-specific plotting utilities for binary classification visualization.

This module provides Plotly-based interactive visualizations for the
SE02 workshop on Artificial Neural Networks, including:
- 2D classification scatter plots with decision boundaries
- Probability field visualizations
- Training loss curves
- Model comparison plots
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Union

__all__ = [
    'plot_2d_classification',
    'plot_training_loss',
    'plot_model_comparison',
    'get_plotly_layout'
]

# Font configuration
FONT_FAMILY = 'Share Tech, monospace'
COLORS = {
    'class_0': '#3498db',  # Blue
    'class_1': '#e74c3c',  # Red
    'boundary': 'black',
    'loss_linear': '#e74c3c',
    'loss_mlp': '#27ae60',
    'grid': 'lightgray'
}


def get_plotly_layout(title: str, width: int = 700, height: int = 600) -> dict:
    """
    Get consistent Plotly layout configuration with Share Tech Mono font.
    
    Args:
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        
    Returns:
        Dictionary of layout parameters
    """
    return dict(
        title=dict(
            text=title,
            font=dict(family=FONT_FAMILY, size=18, color='black')
        ),
        xaxis=dict(
            title=dict(text='Feature x₁', font=dict(family=FONT_FAMILY, size=14)),
            tickfont=dict(family=FONT_FAMILY),
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        yaxis=dict(
            title=dict(text='Feature x₂', font=dict(family=FONT_FAMILY, size=14)),
            tickfont=dict(family=FONT_FAMILY),
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        legend=dict(
            font=dict(family=FONT_FAMILY),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        width=width,
        height=height
    )


def plot_2d_classification(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
    bias: Optional[Union[float, torch.Tensor]] = None,
    title: str = "2D Classification",
    show_boundary: bool = True,
    model: Optional[torch.nn.Module] = None,
    show_probabilities: bool = False,
    width: int = 700,
    height: int = 600
) -> go.Figure:
    """
    Create an interactive Plotly visualization for 2D binary classification.
    
    Args:
        X: Feature matrix, shape (n_samples, 2)
        y: Binary labels (0 or 1), shape (n_samples,)
        weights: Decision boundary weights [w1, w2] for linear models
        bias: Decision boundary bias term
        title: Plot title
        show_boundary: Whether to show decision boundary line (linear models)
        model: Trained PyTorch model for probability contours
        show_probabilities: Whether to show probability field as background
        width: Figure width in pixels
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Convert tensors to numpy
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    # Plot data points for each class
    for class_idx in [0, 1]:
        mask = y == class_idx
        color_key = f'class_{class_idx}'
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {class_idx}',
            marker=dict(
                size=10,
                color=COLORS[color_key],
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            hovertemplate=f'<b>Class {class_idx}</b><br>x₁=%{{x:.2f}}<br>x₂=%{{y:.2f}}<extra></extra>'
        ))
    
    # Add probability field if model is provided
    if show_probabilities and model is not None:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        
        with torch.no_grad():
            if hasattr(model, 'forward'):
                Z = model(grid).numpy()
            else:
                weights_t = torch.FloatTensor(weights) if isinstance(weights, np.ndarray) else weights
                bias_t = torch.FloatTensor([bias]) if isinstance(bias, (int, float)) else bias
                Z = torch.sigmoid(grid @ weights_t.unsqueeze(1) + bias_t).numpy()
        Z = Z.reshape(xx.shape)
        
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 200),
            y=np.linspace(y_min, y_max, 200),
            z=Z,
            colorscale='RdBu_r',
            opacity=0.3,
            showscale=True,
            contours=dict(start=0, end=1, size=0.1),
            colorbar=dict(
                title=dict(text="P(y=1)", font=dict(family=FONT_FAMILY)),
                tickfont=dict(family=FONT_FAMILY)
            ),
            hovertemplate='x₁=%{x:.2f}<br>x₂=%{y:.2f}<br>P(y=1)=%{z:.2f}<extra></extra>',
            name='Probability Field'
        ))
    
    # Add decision boundary for linear models
    if show_boundary and weights is not None and bias is not None:
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        if isinstance(bias, torch.Tensor):
            bias = bias.detach().cpu().numpy()
        if isinstance(bias, np.ndarray):
            bias = bias.item()
        
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x_line = np.linspace(x_min, x_max, 100)
        
        # Decision boundary: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
        if weights[1] != 0:
            y_line = -(weights[0] * x_line + bias) / weights[1]
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name='Decision Boundary',
                line=dict(color=COLORS['boundary'], width=2, dash='dash'),
                hovertemplate='Decision Boundary<br>x₁=%{x:.2f}<br>x₂=%{y:.2f}<extra></extra>'
            ))
    
    # Apply layout
    fig.update_layout(**get_plotly_layout(title, width, height))
    
    return fig


def plot_training_loss(
    losses: List[float],
    title: str = "Training Loss Over Time",
    line_color: str = '#2c3e50',
    width: int = 600,
    height: int = 400
) -> go.Figure:
    """
    Plot training loss curve.
    
    Args:
        losses: List of loss values
        title: Plot title
        line_color: Color for the loss curve
        width: Figure width in pixels
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(losses) + 1)),
        y=losses,
        mode='lines',
        line=dict(color=line_color, width=2),
        name='BCE Loss'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=16)),
        xaxis=dict(
            title=dict(text="Epoch", font=dict(family=FONT_FAMILY)),
            tickfont=dict(family=FONT_FAMILY),
            gridcolor=COLORS['grid']
        ),
        yaxis=dict(
            title=dict(text="Loss", font=dict(family=FONT_FAMILY)),
            tickfont=dict(family=FONT_FAMILY),
            gridcolor=COLORS['grid']
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=width,
        height=height,
        showlegend=False
    )
    
    return fig


def plot_model_comparison(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    linear_weights: Union[np.ndarray, torch.Tensor],
    linear_bias: Union[float, torch.Tensor],
    mlp_model: torch.nn.Module,
    losses_linear: List[float],
    losses_mlp: List[float],
    accuracy_linear: float,
    accuracy_mlp: float,
    width: int = 1400,
    height: int = 450
) -> go.Figure:
    """
    Create side-by-side comparison of linear vs MLP models.
    
    Args:
        X: Feature matrix
        y: Labels
        linear_weights: Weights from linear model
        linear_bias: Bias from linear model
        mlp_model: Trained MLP model
        losses_linear: Training losses for linear model
        losses_mlp: Training losses for MLP model
        accuracy_linear: Final accuracy of linear model
        accuracy_mlp: Final accuracy of MLP model
        width: Figure width in pixels
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object with 3 subplots
    """
    # Convert tensors to numpy
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(linear_weights, torch.Tensor):
        linear_weights = linear_weights.detach().cpu().numpy()
    if isinstance(linear_bias, torch.Tensor):
        linear_bias = linear_bias.detach().cpu().numpy()
    if isinstance(linear_bias, np.ndarray):
        linear_bias = linear_bias.item()
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f'Linear Model (Acc: {accuracy_linear*100:.1f}%)',
            f'MLP Model (Acc: {accuracy_mlp*100:.1f}%)',
            'Training Loss Comparison'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
        horizontal_spacing=0.1
    )
    
    # Plot 1: Linear model with boundary
    for class_idx in [0, 1]:
        mask = y == class_idx
        color = COLORS[f'class_{class_idx}']
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode='markers',
                name=f'Class {class_idx}',
                marker=dict(size=6, color=color, line=dict(width=0.5, color='white')),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Add linear boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_line = np.linspace(x_min, x_max, 100)
    if linear_weights[1] != 0:
        y_line = -(linear_weights[0] * x_line + linear_bias) / linear_weights[1]
        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color=COLORS['boundary'], width=2, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Plot 2: MLP with probability contours
    for class_idx in [0, 1]:
        mask = y == class_idx
        color = COLORS[f'class_{class_idx}']
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode='markers',
                name=f'Class {class_idx}',
                marker=dict(size=6, color=color, line=dict(width=0.5, color='white')),
                showlegend=True,
                legendgroup=str(class_idx)
            ),
            row=1, col=2
        )
    
    # Add MLP probability field
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        Z = mlp_model(grid).numpy()
    Z = Z.reshape(xx.shape)
    
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            colorscale='RdBu_r',
            opacity=0.4,
            showscale=False,
            contours=dict(start=0, end=1, size=0.1),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Plot 3: Loss comparison
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(losses_linear) + 1)),
            y=losses_linear,
            mode='lines',
            name='Linear Model',
            line=dict(color=COLORS['loss_linear'], width=2)
        ),
        row=1, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(losses_mlp) + 1)),
            y=losses_mlp,
            mode='lines',
            name='MLP Model',
            line=dict(color=COLORS['loss_mlp'], width=2)
        ),
        row=1, col=3
    )
    
    # Update axes
    for col in [1, 2]:
        fig.update_xaxes(
            title=dict(text="Feature x₁", font=dict(family=FONT_FAMILY, size=10)),
            tickfont=dict(family=FONT_FAMILY, size=8),
            gridcolor=COLORS['grid'],
            row=1, col=col
        )
        fig.update_yaxes(
            title=dict(text="Feature x₂", font=dict(family=FONT_FAMILY, size=10)),
            tickfont=dict(family=FONT_FAMILY, size=8),
            gridcolor=COLORS['grid'],
            row=1, col=col
        )
    
    fig.update_xaxes(
        title=dict(text="Epoch", font=dict(family=FONT_FAMILY, size=10)),
        tickfont=dict(family=FONT_FAMILY, size=8),
        gridcolor=COLORS['grid'],
        row=1, col=3
    )
    fig.update_yaxes(
        title=dict(text="Loss", font=dict(family=FONT_FAMILY, size=10)),
        tickfont=dict(family=FONT_FAMILY, size=8),
        gridcolor=COLORS['grid'],
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title_text="Linear vs. MLP: A Tale of Two Models",
        title_font=dict(family=FONT_FAMILY, size=16, color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        width=width,
        showlegend=True,
        legend=dict(
            font=dict(family=FONT_FAMILY, size=10),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig
