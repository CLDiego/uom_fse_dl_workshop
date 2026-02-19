import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import FloatSlider, interact, interactive, Layout, interactive_output
from typing import Callable, Optional, Dict, Any, Union, Tuple

from .plots import make_fig_pretty, load_font

__all__ = ["create_interactive_neuron_visualizer",
           "se04_visualize_transformations",
           "wake_cylinder",
           "wake_cylinder_interactive",
           "visualize_flow_comparison",
           "visualize_flow_comparison_interactive"]


def visualize_flow_comparison(
    x_star: np.ndarray,
    u_true: np.ndarray,
    v_true: np.ndarray,
    p_true: np.ndarray,
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    p_pred: np.ndarray,
    time: np.ndarray,
    t_idx: int = 0,
    figsize: tuple = (16, 6),
    axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
    **kwargs: Optional[dict]
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Visualize the comparison between ground truth and predicted flow fields.

    Parameters:
    -----------
    x_star : np.ndarray
        Spatial coordinates of shape (N, 2)
    u_true, v_true : np.ndarray
        True velocity components of shape (N, T)
    p_true : np.ndarray
        True pressure field of shape (N, T)
    u_pred, v_pred : np.ndarray
        Predicted velocity components of shape (N, T)
    p_pred : np.ndarray
        Predicted pressure field of shape (N, T)
    time : np.ndarray
        Time points of shape (T, 1)
    t_idx : int
        Time index to visualize
    figsize : tuple, optional
        Figure size (width, height)
    axes : tuple, optional
        Tuple containing two matplotlib axes to plot on
    **kwargs : dict
        Additional arguments to pass to make_fig_pretty

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object (None if axes is provided)
    axes : tuple
        Tuple containing the two axes objects
    """
    # Reshape grid
    x_unique = np.unique(x_star[:, 0])
    y_unique = np.unique(x_star[:, 1])
    Nx, Ny = len(x_unique), len(y_unique)

    x = x_star[:, 0].reshape(Ny, Nx)
    y = x_star[:, 1].reshape(Ny, Nx)

    # Create figure if needed
    fig = None
    if axes is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        ax1, ax2 = axes

    # Get current timestep data
    u_t = u_true[:, t_idx].reshape(Ny, Nx)
    v_t = v_true[:, t_idx].reshape(Ny, Nx)
    p_t = p_true[:, t_idx].reshape(Ny, Nx)

    u_p = u_pred[:, t_idx].reshape(Ny, Nx)
    v_p = v_pred[:, t_idx].reshape(Ny, Nx)
    p_p = p_pred[:, t_idx].reshape(Ny, Nx)

    # Plot pressure fields
    im1 = ax1.imshow(p_t, extent=[x_unique.min(), x_unique.max(),
                                 y_unique.min(), y_unique.max()],
                     origin='lower', cmap='viridis', aspect='auto')
    im2 = ax2.imshow(p_p, extent=[x_unique.min(), x_unique.max(),
                                 y_unique.min(), y_unique.max()],
                     origin='lower', cmap='viridis', aspect='auto')

    # Add streamplots
    ax1.streamplot(x, y, u_t, v_t, color='white', density=1.5)
    ax2.streamplot(x, y, u_p, v_p, color='white', density=1.5)

    # Add cylinders
    circle1 = plt.Circle((0, 0), 0.1, color='gray', alpha=0.6)
    circle2 = plt.Circle((0, 0), 0.1, color='gray', alpha=0.6)
    ax1.add_patch(circle1)
    ax2.add_patch(circle2)

    # Set axis limits
    for ax in [ax1, ax2]:
        ax.set_xlim(-2, 10)
        ax.set_ylim(-3, 3)

    # Add colorbars with font formatting
    fm = load_font()
    for ax, im, title in zip([ax1, ax2], [im1, im2], 
                            ['Ground Truth', 'PINN Prediction']):
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('PRESSURE', fontproperties=fm)
        cbar.ax.tick_params(labelsize=11)
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(fm)
        
        make_fig_pretty(
            ax=ax,
            title=f"{title} (t = {time[t_idx, 0]:.2f}s)",
            xlabel="X",
            ylabel="Y",
            legend=False,
            grid=False,
            **kwargs
        )

    if fig is not None:
        plt.tight_layout()

    return fig, (ax1, ax2)

def visualize_flow_comparison_interactive(
    x_star, u_true, v_true, p_true, 
    u_pred, v_pred, p_pred, time, 
    figsize=(16, 6)
):
    output = widgets.Output()

    @widgets.interact(t_idx=widgets.IntSlider(
        min=0, max=len(time)-1, step=1, description='Time Step'))
    def update(t_idx):
        with output:
            output.clear_output(wait=True)
            fig, _ = visualize_flow_comparison(
                x_star, u_true, v_true, p_true,
                u_pred, v_pred, p_pred, time,
                t_idx=t_idx, figsize=figsize
            )
            plt.show()
            plt.close(fig)

    display(output)

def wake_cylinder(
    X_star: np.array,
    u_star: np.array,
    p_star: np.array,
    time: np.array,
    t_idx: int = 0,
    figsize: tuple = (8, 6),
    ax: Optional[plt.Axes] = None,
    **kwargs: Optional[dict]
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the wake behind a cylinder at a specific time step.

    Parameters:
    -----------
    X_star : np.ndarray
        Spatial coordinates of shape (N, 2)
    u_star : np.ndarray
        Velocity components of shape (N, 2, T)
    p_star : np.ndarray
        Pressure field of shape (N, T)
    time : np.ndarray
        Time points of shape (T, 1)
    t_idx : int
        Time index to visualize
    figsize : tuple, optional
        Figure size (width, height)
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    **kwargs : dict
        Additional arguments to pass to make_fig_pretty

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object (None if ax is provided)
    ax : matplotlib.axes.Axes
        Axes object
    """
    # Reshape grid
    x_unique = np.unique(X_star[:, 0])
    y_unique = np.unique(X_star[:, 1])
    Nx, Ny = len(x_unique), len(y_unique)

    x = X_star[:, 0].reshape(Ny, Nx)
    y = X_star[:, 1].reshape(Ny, Nx)

    # Create figure if needed
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Get current timestep data
    u = u_star[:, 0, t_idx].reshape(Ny, Nx)
    v = u_star[:, 1, t_idx].reshape(Ny, Nx)
    p = p_star[:, t_idx].reshape(Ny, Nx)

    # Add cylinder
    circle = plt.Circle((0, 0), 0.1, color='gray', alpha=0.6)
    ax.add_patch(circle)

    # set axis limits
    ax.set_xlim(-2, 10)
    ax.set_ylim(-3, 3)

    # Plot pressure field
    im = ax.imshow(p, extent=[x_unique.min(), x_unique.max(),
                              y_unique.min(), y_unique.max()],
                   origin='lower', cmap='viridis', aspect='auto')

    # Add streamplot
    ax.streamplot(x, y, u, v, color='white', density=1.5)

    # Add colorbar with font formatting
    fm = load_font()
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PRESSURE', fontproperties=fm)
    cbar.ax.tick_params(labelsize=11)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(fm)

    # Make plot pretty
    make_fig_pretty(
        ax=ax,
        title=f"Time: {time[t_idx, 0]:.2f} s",
        xlabel="X",
        ylabel="Y",
        legend=False,
        grid=False,
        **kwargs
    )

    if fig is not None:
        plt.tight_layout()

    return fig, ax

def wake_cylinder_interactive(X_star, u_star, p_star, time, figsize=(8, 6)):
    """
    Create an interactive version of the wake cylinder visualization
    using IPython widgets.
    """
    output = widgets.Output()

    @widgets.interact(t_idx=widgets.IntSlider(
        min=0, max=len(time)-1, step=1, description='Time Step'))
    def update(t_idx):
        with output:
            output.clear_output(wait=True)
            fig, _ = wake_cylinder(X_star, u_star, p_star, time,
                                   t_idx=t_idx, figsize=figsize)
            plt.show()
            plt.close(fig) 

    display(output)

def _update_plot(
    w: float,
    b: float,
    X: torch.Tensor,
    y: torch.Tensor,
    X_mean: float,
    X_std: float,
    challenger_temp: float,
    neuron_class: type,
    loss_function: Callable,
    out: widgets.Output
) -> None:
    """
    Internal function to update the visualization plot based on slider values.

    Args:
        w: Weight parameter value
        b: Bias parameter value
        X: Feature tensor
        y: Target tensor
        X_mean: Mean of the features (for normalization)
        X_std: Standard deviation of the features (for normalization)
        challenger_temp: The Challenger launch temperature
        neuron_class: The class used to create the neuron model
        loss_function: Function to compute the loss
        out: Output widget to display the plot
    """
    with out:
        out.clear_output()

        # Create a neuron with the specified parameters
        custom_model = neuron_class(n_features=1)
        with torch.no_grad():
            # Set the weights and bias to the specified values
            custom_model.weights.copy_(torch.tensor([w]))
            custom_model.bias.copy_(torch.tensor(b))

        # Calculate predictions
        predictions = torch.zeros_like(y)
        X_normalized = X  # Assuming X is already normalized

        for i in range(len(X_normalized)):
            predictions[i] = custom_model(X_normalized[i])

        # Calculate loss
        loss = loss_function(predictions, y)

        # Convert tensors to numpy for plotting
        temperatures = X.numpy().flatten() * X_std + X_mean  # De-normalize for plotting
        actual = y.numpy().flatten()

        # Plot the results
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot data points
        ax.scatter(temperatures, actual, color='None', edgecolor='k',
                   s=100, alpha=0.7, label='Actual Data')

        # Plot model predictions
        temp_range = np.linspace(
            min(temperatures) - 2, max(temperatures) + 2, 100)
        temp_range_normalized = (temp_range - X_mean) / X_std
        smooth_preds = np.array(
            [custom_model(torch.tensor([t]).float()).item() for t in temp_range_normalized])
        ax.plot(temp_range, smooth_preds, 'orange',
                linewidth=3, label='Model Prediction')

        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=challenger_temp, color='black', linestyle='--', alpha=0.7)
        ax.text(challenger_temp+0.5, 0.8,
                f'Challenger Launch: {challenger_temp:.1f}°C', rotation=90)

        # Calculate error visually
        for i, (temp, true_val, pred_val) in enumerate(zip(temperatures, actual, predictions.detach().numpy())):
            if abs(true_val - pred_val.item()) > 0.05:  # Only show significant errors
                ax.plot([temp, temp], [true_val, pred_val.item()],
                        'b-', alpha=0.3)

        make_fig_pretty(
            ax=ax,
            title=f"O-Ring Failure Model (Loss: {loss.item():.4f})",
            xlabel="Temperature (°C)",
            ylabel="Probability of Failure",
            legend=True,
            legd_loc='upper right',
            grid=True
        )

        plt.tight_layout()
        plt.show()

def _train_model_callback(
    b: widgets.Button,
    w_slider: widgets.FloatSlider,
    b_slider: widgets.FloatSlider,
    X: torch.Tensor,
    y: torch.Tensor,
    neuron_class: type,
    loss_function: Callable,
    learning_rate: float = 0.1,
    epochs: int = 100
) -> None:
    """
    Callback function to train the model and update sliders.

    Args:
        b: Button widget that triggered the callback
        w_slider: Slider for weight parameter
        b_slider: Slider for bias parameter
        X: Feature tensor
        y: Target tensor
        neuron_class: The class used to create the neuron model
        loss_function: Function to compute the loss
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
    """
    # Reset model
    new_model = neuron_class(n_features=1)

    # Train for specified epochs
    optimizer = torch.optim.SGD(
        [new_model.weights, new_model.bias], lr=learning_rate)
    for _ in range(epochs):
        # Forward pass
        predictions = torch.zeros_like(y)
        for i in range(len(X)):
            predictions[i] = new_model(X[i])

        # Compute loss
        loss = loss_function(predictions, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update sliders to the trained values
    w_slider.value = float(new_model.weights.item())
    b_slider.value = float(new_model.bias.item())

def create_interactive_neuron_visualizer(
    X: torch.Tensor,
    y: torch.Tensor,
    X_mean: float,
    X_std: float,
    challenger_temp: float,
    neuron_class: type,
    loss_function: Callable,
    initial_weight: Optional[float] = None,
    initial_bias: Optional[float] = None,
    learning_rate: float = 0.1,
    training_epochs: int = 100
) -> widgets.Widget:
    """
    Create an interactive visualization widget for exploring neuron parameters.

    Args:
        X: Feature tensor (normalized)
        y: Target tensor
        X_mean: Mean of the features (before normalization)
        X_std: Standard deviation of the features (before normalization)
        challenger_temp: The Challenger launch temperature
        neuron_class: The class used to create the neuron model
        loss_function: Function to compute the loss
        initial_weight: Initial weight value for the slider (default uses model's weight)
        initial_bias: Initial bias value for the slider (default uses model's bias)
        learning_rate: Learning rate for the training button
        training_epochs: Number of epochs when training button is clicked

    Returns:
        A widget containing sliders and a train button for interactive visualization
    """
    # Create a temporary model to get default parameter values if not provided
    if initial_weight is None or initial_bias is None:
        temp_model = neuron_class(n_features=1)
        if initial_weight is None:
            try:
                initial_weight = float(temp_model.weights.item())
            except:
                initial_weight = 0.0
        if initial_bias is None:
            try:
                initial_bias = float(temp_model.bias.item())
            except:
                initial_bias = 0.0

    # Create output widget and parameter sliders
    out = widgets.Output()
    w_slider = FloatSlider(value=initial_weight, min=-
                           10.0, max=10.0, step=0.1, description='Weight:')
    b_slider = FloatSlider(value=initial_bias, min=-10.0,
                           max=10.0, step=0.1, description='Bias:')
    train_button = widgets.Button(description="Train Model")

    # Wrap update_plot with fixed parameters
    def wrapped_update_plot(w, b):
        _update_plot(w, b, X, y, X_mean, X_std, challenger_temp,
                     neuron_class, loss_function, out)

    # Connect callbacks
    train_button.on_click(lambda b: _train_model_callback(
        b, w_slider, b_slider, X, y, neuron_class, loss_function, learning_rate, training_epochs
    ))

    # Create the interactive widget
    inter_out = interactive_output(
        wrapped_update_plot, {'w': w_slider, 'b': b_slider})

    # Combine all widgets
    main_ui = widgets.VBox([
        widgets.HBox([w_slider, b_slider, train_button]),
        out
    ])

    # Initial update
    wrapped_update_plot(w_slider.value, b_slider.value)

    return main_ui, inter_out

def se04_visualize_transformations(transformed_images):
    """Visualize the transformed images.

    Args:
        transformed_images (dict): Dictionary of transformed images
    """
    _, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes = axes.flatten()

    for i, (title, img_tensor) in enumerate(transformed_images.items()):
        if title == 'Normalized':
            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean

        img_numpy = img_tensor.permute(1, 2, 0).numpy()
        img_numpy = np.clip(img_numpy, 0, 1)

        axes[i].imshow(img_numpy)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
