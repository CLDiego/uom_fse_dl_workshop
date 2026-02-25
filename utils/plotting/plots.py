import scipy
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from .formatting import make_fig_pretty
import networkx as nx
import matplotlib.patches as mpatches
from .fonts import load_font
from matplotlib.lines import Line2D


__all__ = ["plot_loss", "plot_distribution",
           "visualize_network_nx", "plot_model_predictions_SE02",
           "show_model_predictions",
           "show_binary_segmentation_batch",
           "show_binary_segmentation_predictions",
           "compare_binary_segmentation_models"]


def plot_loss(
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    title: str = "Training Progress",
    figsize: tuple = (10, 6),
    ax: Optional[plt.Axes] = None,
    **kwargs: Optional[dict]
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot training and validation loss curves.

    Args:
        train_loss: List of training loss values
        val_loss: Optional list of validation loss values
        title: Plot title
        figsize: Figure size as (width, height)
        ax: Optional matplotlib axes to plot on
        **kwargs: Additional arguments to pass to make_fig_pretty

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_loss) + 1)

    # Plot training loss
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')

    # Plot validation loss if provided
    if val_loss is not None:
        ax.plot(epochs, val_loss, 'r--', linewidth=2, label='Validation Loss')

    make_fig_pretty(
        ax=ax,
        xlabel="Epoch",
        ylabel="Loss",
        title=title,
        legend=True,
        legd_loc='upper right',
        grid=True,
        **kwargs
    )

    if fig is not None:
        plt.tight_layout()

    return fig, ax


def plot_distribution(
        data: List[float],
        ax: plt.axes,
        title: str = "Distribution Plot",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        bins: int = 30,
        kdensity: bool = False,
        make_pretty: bool = True,
        **kwargs: Optional[dict]) -> None:

    sharex = kwargs.get('sharex', False)
    sharey = kwargs.get('sharey', False)
    kernel = scipy.stats.gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 1000)
    kde = kernel(x_range)

    ax.hist(data, bins=bins, density=True, alpha=0.5, color='skyblue')

    if kdensity:
        ax.plot(x_range, kde, color='red', linewidth=2, label='KDE')

    if make_pretty:
        make_fig_pretty(
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=True,
            legd_loc='upper right',
            grid=False,
            xtick_fsize=10,
            ytick_fsize=10,
            xlabel_fsize=10,
            ylabel_fsize=10,
            title_fsize=10,
            sharex=sharex,
            sharey=sharey
        )
        return
    else:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax


def visualize_network_nx(model: torch.nn.Module, figsize: tuple = (8, 4)) -> Tuple[plt.Figure, plt.Axes, nx.DiGraph, Dict[str, Any]]:
    """
    Visualize a neural network architecture using networkx.

    Args:
        model: PyTorch neural network model
        figsize: Figure size as (width, height)

    Returns:
        fig, ax, G, pos: Matplotlib figure, axes, networkx graph, and node positions
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Extract model layers and their dimensions
    layers = []
    layer_names = []
    activations = []

    # Process all modules in the model
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            layers.append({
                'type': 'linear',
                'in_features': module.in_features,
                'out_features': module.out_features,
                'name': name
            })
            layer_names.append(name)
        elif any(isinstance(module, act_type) for act_type in [
            torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Sigmoid,
            torch.nn.Tanh, torch.nn.Softmax, torch.nn.GELU
        ]):
            activations.append(type(module).__name__)

    # For simple models or manually created ones, extract parameters
    if not layers and hasattr(model, 'weights'):
        try:
            in_features = model.weights.shape[0] if len(
                model.weights.shape) == 1 else model.weights.shape[1]
            out_features = 1
            layers.append({
                'type': 'custom',
                'in_features': in_features,
                'out_features': out_features,
                'name': 'custom_layer'
            })
            layer_names.append('custom_layer')
            if hasattr(model, 'activation'):
                activations.append(model.activation.__name__ if hasattr(
                    model.activation, '__name__') else 'Custom')
        except (AttributeError, IndexError):
            # Fall back to a simple representation if structure can't be determined
            in_features = 1
            out_features = 1

    # If still no layers found, create a placeholder
    if not layers:
        print("Warning: Could not determine network structure automatically")
        return None, None, None, None

    # Simplify the network representation
    # Create basic nodes: inputs, hidden layer, and output
    input_nodes = [f"x{i+1}" for i in range(layers[0]['in_features'])]
    hidden_nodes = ["Neuron"] if layers[0]['out_features'] == 1 else [
        f"h{i+1}" for i in range(layers[0]['out_features'])]
    output_nodes = ["y"] if len(layers) == 1 or layers[-1]['out_features'] == 1 else [
        f"y{i+1}" for i in range(layers[-1]['out_features'])]

    # Add all nodes to the graph
    for node in input_nodes + hidden_nodes + output_nodes:
        G.add_node(node)

    # Add edges from inputs to hidden layer
    for input_node in input_nodes:
        for hidden_node in hidden_nodes:
            G.add_edge(input_node, hidden_node)

    # If we have a deeper network, add edges from hidden to output
    if len(layers) > 1:
        for hidden_node in hidden_nodes:
            for output_node in output_nodes:
                G.add_edge(hidden_node, output_node)
    else:
        # Add direct edges from hidden to output for single layer networks
        for hidden_node in hidden_nodes:
            for output_node in output_nodes:
                G.add_edge(hidden_node, output_node)

    # Create a simple, clean layout
    pos = {}

    # Position input nodes vertically on the left
    input_spacing = 2 / max(len(input_nodes), 1)
    for i, node in enumerate(input_nodes):
        pos[node] = (-1, (i * input_spacing) - 1 + input_spacing/2)

    # Position hidden nodes in the middle
    hidden_spacing = 2 / max(len(hidden_nodes), 1)
    for i, node in enumerate(hidden_nodes):
        pos[node] = (0, (i * hidden_spacing) - 1 + hidden_spacing/2)

    # Position output nodes on the right
    output_spacing = 2 / max(len(output_nodes), 1)
    for i, node in enumerate(output_nodes):
        pos[node] = (1, (i * output_spacing) - 1 + output_spacing/2)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Load font for consistent styling - use the same font for everything
    fm = load_font()

    # Define pleasing colors
    node_color = '#AED6F1'  # Light blue
    edge_color = '#2C3E50'  # Dark blue/gray
    activation_color = '#E74C3C'  # Red for activation functions

    # Draw nodes first (so edges appear on top)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=2000,
        node_color=node_color,
        edgecolors='black',
        linewidths=1.5,
        ax=ax
    )

    # Identify activation edges - always the connections to output layer
    act_edges = []
    regular_edges = []

    for u, v in G.edges():
        if v in output_nodes:
            act_edges.append((u, v))
        else:
            regular_edges.append((u, v))

    # Draw regular edges with straight lines for better arrow visibility
    nx.draw_networkx_edges(
        G, pos,
        edgelist=regular_edges,
        arrowsize=20,
        width=1.5,
        alpha=0.7,
        edge_color=edge_color,
        connectionstyle='arc3,rad=0.0',  # Straight lines for clarity
        arrows=True,  # Ensure arrows are shown
        arrowstyle='-|>',  # Clear arrow style
        ax=ax
    )

    # Draw activation function edges with different style and color
    if activations and act_edges:
        # Get activation name for display in legend
        act_name = ", ".join(set(activations))

        # Draw activation edges with distinctive style
        nx.draw_networkx_edges(
            G, pos,
            edgelist=act_edges,
            arrowsize=25,
            width=3.0,
            alpha=1.0,
            edge_color=activation_color,
            connectionstyle='arc3,rad=0.0',  # Straight lines for better arrows
            style='solid',
            arrows=True,
            arrowstyle='-|>',  # More visible arrow style
            min_source_margin=15,  # Give space for the arrow to be visible
            min_target_margin=15,
            ax=ax
        )

        # We no longer add the dot on the activation function lines

    # Draw node labels with consistent font
    label_font = fm.copy()
    label_font.set_weight('bold')

    # Custom draw labels with consistent font
    for node, (x, y) in pos.items():
        ax.text(x, y, node,
                fontproperties=label_font,
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='center',
                color='black',
                transform=ax.transData,
                zorder=12)

    # Add a title with network size information
    n_input = len(input_nodes)
    n_hidden = len(hidden_nodes)
    n_output = len(output_nodes)
    title = f"Neural Network ({n_input}-{n_hidden}-{n_output})"
    if activations:
        title += "\nActivations: " + ", ".join(set(activations))

    # Add legend for activation function if present
    legend_elements = [
        mpatches.Patch(color=node_color, label='Network Nodes'),
        Line2D([0], [0], color=edge_color, lw=1.5, label='Connections')
    ]

    if activations:
        legend_elements.append(
            Line2D([0], [0], color=activation_color, lw=3.0,
                   label=f'{act_name} Activation')  # Removed marker
        )

    # Create the legend with consistent font in the top-left position
    legend = ax.legend(
        handles=legend_elements,
        loc='upper left',  # Changed to upper left as requested
        prop=fm,
        framealpha=0.8  # Better visibility of legend
    )

    # Make figure pretty with consistent font throughout
    make_fig_pretty(
        ax=ax,
        title=title,
        grid=False
    )

    # Make sure there's enough padding around the graph
    plt.tight_layout(pad=1.2)

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    return fig, ax, G, pos


def plot_model_predictions_SE02(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    X_mean: float,
    X_std: float,
    special_temp: Optional[float] = None,
    special_temp_label: Optional[str] = None,
    title: str = "Model Predictions",
    figsize: tuple = (10, 6),
    ax: Optional[plt.Axes] = None,
    **kwargs: Optional[dict]
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot model predictions against actual data points.

    Args:
        model: The trained model
        X: Feature tensor (normalized)
        y: Target tensor
        X_mean: Mean value used for normalization
        X_std: Standard deviation used for normalization
        special_temp: Optional temperature to highlight with a vertical line
        special_temp_label: Label for the special temperature
        title: Plot title
        figsize: Figure size as (width, height)
        ax: Optional matplotlib axes to plot on
        **kwargs: Additional arguments to pass to make_fig_pretty

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Convert tensors to numpy for plotting
    temperatures = X.numpy().flatten() * X_std + X_mean  # De-normalize for plotting
    actual = y.numpy().flatten()

    # Create temperature range for smooth curve
    temp_range = np.linspace(min(temperatures) - 2, max(temperatures) + 2, 100)
    temp_range_normalized = (temp_range - X_mean) / X_std

    # Get predictions for the whole temperature range
    with torch.no_grad():
        smooth_preds = np.array(
            [model(torch.tensor([t]).float()).item() for t in temp_range_normalized])

    # Plot the data points and model prediction curve
    ax.scatter(temperatures, actual, color='None', edgecolor='black',
               s=100, alpha=0.7, label='Actual Data')
    ax.plot(temp_range, smooth_preds, 'orange',
            linewidth=2, label='Model Prediction')

    # Add threshold line at 0.5 probability
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)

    # Add special temperature line if provided
    if special_temp is not None:
        ax.axvline(x=special_temp, color='black', linestyle='--', alpha=0.7)
        if special_temp_label:
            ax.text(special_temp+0.5, 0.8, special_temp_label, rotation=90)

    make_fig_pretty(
        ax=ax,
        title=title,
        xlabel="Temperature (°C)",
        ylabel="Probability of Failure",
        legend=True,
        legd_loc='upper right',
        grid=True,
        **kwargs
    )

    if fig is not None:
        plt.tight_layout()

    return fig, ax


def show_model_predictions(model, data_loader, class_names, num_images=12, title=None):
    """
    Visualize model predictions on a dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate
        data_loader (DataLoader): DataLoader containing the dataset to visualize
        class_names (list): List of class names
        num_images (int, optional): Number of images to display. Defaults to 12.
        title (str, optional): Title for the visualization. Defaults to None.

    Returns:
        None: Displays a matplotlib figure with predictions
    """
    # Set device for model evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Initialize figure for visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Collect predictions on a larger batch to get accuracy statistics
    all_samples = []
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Track overall accuracy
            batch_correct = (preds == labels).sum().item()
            total_correct += batch_correct
            total_samples += labels.size(0)
            
            # Collect samples for display
            for j in range(inputs.size(0)):
                is_correct = preds[j] == labels[j]
                
                # Move tensors to CPU for numpy conversion
                sample = {
                    'image': inputs[j].cpu().permute(1, 2, 0).numpy(),
                    'pred': preds[j].cpu().item(),
                    'true': labels[j].cpu().item(),
                    'is_correct': is_correct
                }
                
                all_samples.append(sample)
                
            if len(all_samples) >= num_images * 3:  # Collect more than we need
                break
    
    # Get overall accuracy
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Randomly select samples for display
    if len(all_samples) > num_images:
        display_samples = random.sample(all_samples, num_images)
    else:
        display_samples = all_samples
    
    # Plot the samples
    for i, sample in enumerate(display_samples):
        ax = fig.add_subplot(3, 4, i + 1)
        
        # Display image (clipping to [0,1] range for proper display)
        img_display = np.clip(sample['image'], 0, 1)
        
        # Handle grayscale images (1 channel) vs RGB (3 channels)
        if img_display.ndim == 3 and img_display.shape[2] == 1:
            ax.imshow(img_display.squeeze(2), cmap='gray')
        else:
            ax.imshow(img_display)
        
        pred_class = class_names[sample['pred']]
        true_class = class_names[sample['true']]
        
        title_text = f'Pred: {pred_class}\nTrue: {true_class}'
        title_color = 'green' if sample['is_correct'] else 'red'
        
        # Use make_fig_pretty for consistent styling
        make_fig_pretty(
            ax=ax,
            title=title_text,
            title_color=title_color,
            title_fsize=11,
            grid=False,
            is_image=True
        )
    
    if title:
        fig_title = f"{title} - Overall Accuracy: {overall_accuracy:.1%}"
        plt.suptitle(fig_title, fontsize=16)
    # else:
    #     fig_title = f"Model Predictions - Overall Accuracy: {overall_accuracy:.1%}"
    

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Print overall statistics
    correct_count = sum(1 for s in display_samples if s['is_correct'])
    print(f"Displayed samples: {correct_count}/{num_images} correct ({correct_count/num_images:.1%})")

def show_binary_segmentation_batch(dl: torch.utils.data.DataLoader, n_images: int = 10, mean: torch.Tensor = None, std: torch.Tensor = None):
    """
    Display a batch of images and their corresponding segmentation masks.

    Args:
        dl: DataLoader containing the dataset
        n_images: Number of images to display
    """

    if mean is None or std is None:
        # Default to ones
        mean = torch.ones(3)
        std = torch.ones(3)

    for images, masks in dl:
        _, ax = plt.subplots(nrows=2, ncols=n_images, figsize=(20, 5))

        for ix in range(n_images):
            # Denormalize the image
            img = images[ix].permute(1, 2, 0).numpy()
            img = img * std.numpy() + mean.numpy()  # Denormalize
            img = np.clip(img, 0, 1)  # Clip values to valid range

            ax[0, ix].imshow(img)
            ax[0, ix].axis('off')
            make_fig_pretty(ax=ax[0, ix], grid=False,
                            title=f"Image {ix+1}",
                            is_image=True,)

            ax[1, ix].imshow(masks[ix].squeeze(), cmap='gray')
            ax[1, ix].axis('off')
            make_fig_pretty(ax=ax[1, ix],
                            title=f"Mask {ix+1}",
                            is_image=True)

        plt.show()
        break


def show_binary_segmentation_predictions(model: torch.nn.Module,
                                         dl: torch.utils.data.DataLoader,
                                         n_images=10,
                                         mean: torch.Tensor = None,
                                         std: torch.Tensor = None) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mean is None or std is None:
        # Default to ones
        mean = torch.ones(3)
        std = torch.ones(3)

    model.eval()
    with torch.no_grad():
        for images, masks in dl:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # Apply threshold for binary segmentation
            outputs = (outputs > 0.5).float()

            _, ax = plt.subplots(nrows=3, ncols=n_images, figsize=(20, 5))
            for ix in range(n_images):
                # Denormalize the image
                img = images[ix].permute(1, 2, 0).cpu().numpy()
                img = img * std.cpu().numpy() + mean.cpu().numpy()
                img = np.clip(img, 0, 1)

                ax[0, ix].imshow(img)
                ax[0, ix].axis('off')
                make_fig_pretty(ax=ax[0, ix],
                                title=f"Image {ix+1}",
                                is_image=True)

                ax[1, ix].imshow(masks[ix].squeeze().cpu(), cmap='gray')
                ax[1, ix].axis('off')
                make_fig_pretty(ax=ax[1, ix],
                                title=f"GT {ix+1}",
                                is_image=True)

                ax[2, ix].imshow(outputs[ix].squeeze().cpu(), cmap='gray')
                ax[2, ix].axis('off')
                make_fig_pretty(ax=ax[2, ix],
                                title=f"PR {ix+1}",
                                is_image=True)

            plt.show()
            break


def compare_binary_segmentation_models(model1: torch.nn.Module,
                                       model2: torch.nn.Module,
                                       dl: torch.utils.data.DataLoader,
                                       n_images=10,
                                       mean: torch.Tensor = None,
                                       std: torch.Tensor = None) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mean is None or std is None:
        # Default to ones
        mean = torch.ones(3)
        std = torch.ones(3)

    model1.eval()
    model2.eval()

    with torch.no_grad():
        for images, masks in dl:
            images = images.to(device)
            masks = masks.to(device)

            outputs1 = model1(images)
            outputs2 = model2(images)

            # Apply threshold for binary segmentation
            outputs1 = (outputs1 > 0.5).float()
            outputs2 = (outputs2 > 0.5).float()

            _, ax = plt.subplots(nrows=4, ncols=n_images, figsize=(20, 5))
            for ix in range(n_images):
                # Denormalize the image
                img = images[ix].permute(1, 2, 0).cpu().numpy()
                img = img * std.cpu().numpy() + mean.cpu().numpy()
                img = np.clip(img, 0, 1)

                ax[0, ix].imshow(img)
                ax[0, ix].axis('off')
                make_fig_pretty(ax=ax[0, ix],
                                title=f"Image {ix+1}",
                                is_image=True)

                ax[1, ix].imshow(masks[ix].squeeze().cpu(), cmap='gray')
                ax[1, ix].axis('off')
                make_fig_pretty(ax=ax[1, ix],
                                title=f"GT {ix+1}",
                                is_image=True)

                ax[2, ix].imshow(outputs1[ix].squeeze().cpu(), cmap='gray')
                ax[2, ix].axis('off')
                make_fig_pretty(ax=ax[2, ix],

                                title=f"M1 PR {ix+1}",
                                is_image=True)

                ax[3, ix].imshow(outputs2[ix].squeeze().cpu(), cmap='gray')
                ax[3, ix].axis('off')
                make_fig_pretty(ax=ax[3, ix],
                                title=f"M2 PR {ix+1}",
                                is_image=True)

            plt.show()
            break
