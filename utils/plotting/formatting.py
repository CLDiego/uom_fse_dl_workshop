from typing import List, Optional
from matplotlib.axes import Axes
from utils.plotting.fonts import load_font

__all__ = [
    'format_tick_label',
    'format_axis_ticks',
    'format_spines',
    'make_fig_pretty'
]

PATTERNS = [
    "/", "\\", "|", "-", "o", "O", ".", "*", "+", "x", "X", ":", "=", " ", " "
]

def format_tick_label(label: str) -> str:
    """
    Format tick label text to uppercase if it's a string, leave numbers unchanged.
    
    Args:
        label (str): The tick label to format
        
    Returns:
        str: Formatted tick label
    """
    try:
        if str(label).replace(".", "").replace("-", "").isdigit():
            return str(label)
        return str(label).upper()
    except AttributeError:
        return ""

def format_axis_ticks(
    ax: Axes,
    axis: str,
    share_axis: bool,
    is_3d: bool,
    font_manager,
    tick_size: int
) -> None:
    """
    Format ticks for a given axis (x, y, or z).
    """
    # Set ticks
    getattr(ax, f'set_{axis}ticks')(getattr(ax, f'get_{axis}ticks')())
    
    # Check if axis is log scale
    is_log = getattr(ax, f'get_{axis}scale')() == 'log'
    
    # Handle shared axes
    if share_axis and not is_3d and axis in ['x', 'y']:
        getattr(ax, f'set_{axis}ticklabels')([''] * len(getattr(ax, f'get_{axis}ticklabels')()))
        return
        
    # Format labels if they exist
    labels = [label.get_text() for label in getattr(ax, f'get_{axis}ticklabels')()]
    if all(labels):
        if is_log:
            # For log scale, keep original formatting
            labels_fmt = labels
        else:
            labels_fmt = [format_tick_label(label) for label in labels]
        getattr(ax, f'set_{axis}ticklabels')(labels_fmt)
        
    # Apply font properties
    for label in getattr(ax, f'get_{axis}ticklabels')():
        label.set_fontproperties(font_manager)
        label.set_fontsize(tick_size)

def format_spines(ax: Axes, ylabels_fmt: Optional[List[str]], sharey: bool, timestamp: bool, tufte_style: bool = False) -> None:
    """
    Format plot spines (borders).
    """
    if tufte_style:
        # In Tufte style, remove all spines except where needed for data context
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(False)
        
        # Add subtle ticks without lines
        ax.tick_params(axis='both', which='both', length=3, width=0.5, 
                       colors='gray', labelcolor='black', pad=3)
        return
        
    # Standard formatting (non-Tufte)
    # Remove top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    if timestamp:
        return
        
    # Set spine bounds
    x_bounds = ax.get_xticks()
    y_bounds = ax.get_yticks()
    
    # Check if axes are log scale
    is_xlog = ax.get_xscale() == 'log'
    is_ylog = ax.get_yscale() == 'log'
    
    if not is_xlog:
        ax.spines['bottom'].set_bounds(min(x_bounds), max(x_bounds) * 1.025)
    
    # Handle y-axis spine bounds based on data characteristics
    if (ylabels_fmt or sharey) and not is_ylog and not (ylabels_fmt[0].replace(".", "").isdigit()):
        ax.spines['left'].set_bounds(
            min(y_bounds) + 0.025 * min(y_bounds),
            max(y_bounds) * 1.15
        )
    elif not is_ylog:
        if min(y_bounds) < 0:
            ax.spines['left'].set_bounds(min(y_bounds) * 0.85, max(y_bounds) * 1.05)
            ax.set_yticks(ax.get_yticks()[1:])
        else:
            ax.spines['left'].set_bounds(
                min(y_bounds) * (1.05 if min(y_bounds) == 0 else 1.025),
                max(y_bounds) * 1.025
            )
            ax.set_yticks(ax.get_yticks()[1:])

def make_fig_pretty(
    ax: Axes,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    legend: bool = True,
    grid: bool = False,
    **kwargs
) -> None:
    """
    Apply consistent formatting to a matplotlib figure with Tufte principles option.
    
    Args:
        ax: Matplotlib axis object
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        legend: Whether to show legend
        grid: Whether to show grid
        **kwargs: Additional arguments including:
            zlabel: Z-axis label (for 3D plots)
            ctab: Whether to add patterns to cross-tables
            legd_title: Legend title 
            legd_loc: Legend location
            legd_labels: Custom legend labels
            sharex: Whether x-axis is shared
            sharey: Whether y-axis is shared
            timestamp: Whether to include timestamp
            xtick_fsize: X-axis tick font size (default: 11)
            ytick_fsize: Y-axis tick font size (default: 11)
            xlabel_fsize: X-axis label font size (default: 11)
            ylabel_fsize: Y-axis label font size (default: 11)
            title_fsize: Title font size (default: 12)
            tufte_style: Whether to apply Tufte-style formatting
            background_color: Optional custom background color
            range_frame: Whether to use range frame
            is_image: Whether plot contains images
    """
    # Extract optional parameters with defaults
    zlabel = kwargs.get('zlabel', '')
    ctab = kwargs.get('ctab', False)
    legd_title = kwargs.get('legd_title', '')
    legd_loc = kwargs.get('legd_loc', 'best')
    legd_labels = kwargs.get('legd_labels', None)
    sharex = kwargs.get('sharex', False)
    sharey = kwargs.get('sharey', False)
    timestamp = kwargs.get('timestamp', False)
    xtick_fsize = kwargs.get('xtick_fsize', 11)
    ytick_fsize = kwargs.get('ytick_fsize', 11)
    xlabel_fsize = kwargs.get('xlabel_fsize', 11)
    ylabel_fsize = kwargs.get('ylabel_fsize', 11)
    title_fsize = kwargs.get('title_fsize', 12)
    tufte_style = kwargs.get('tufte_style', False)
    background_color = kwargs.get('background_color', None)
    range_frame = kwargs.get('range_frame', False)
    is_image = kwargs.get('is_image', False)

    is_3d = "3D" in str(type(ax.axes))
    
    # Load and configure fonts
    fm = load_font()
    fm.set_size(9)
    
    # Set background color if specified
    if background_color:
        ax.set_facecolor(background_color)
        
    # Set title and axis labels with special handling for images
    if is_image:
        if tufte_style:
            ax.set_title(title, fontproperties=fm, fontsize=title_fsize-1, 
                         color='#505050', loc='left', pad=8)
            ax.axis('off')
        else:
            title_text = " ".join(title.upper())
            ax.set_title(title_text, fontproperties=fm, fontsize=title_fsize, 
                         loc='left', pad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        if xlabel:
            ax.set_xlabel(xlabel if tufte_style else " ".join(xlabel.upper()),
                       fontproperties=fm, fontsize=xlabel_fsize)
        if ylabel:
            ax.set_ylabel(ylabel if tufte_style else " ".join(ylabel.upper()),
                       fontproperties=fm, fontsize=ylabel_fsize)
    else:
        if tufte_style:
            ax.set_title(title, fontproperties=fm, fontsize=title_fsize-1, color='#505050')
            ax.set_xlabel(xlabel, fontproperties=fm, fontsize=xlabel_fsize, color='#505050')
            ax.set_ylabel(ylabel, fontproperties=fm, fontsize=ylabel_fsize, color='#505050')
            if is_3d:
                ax.set_zlabel(zlabel, fontproperties=fm, fontsize=11, color='#505050')
        else:
            ax.set_title(" ".join(title.upper()), fontproperties=fm, fontsize=title_fsize)
            ax.set_xlabel(" ".join(xlabel.upper()), fontproperties=fm, fontsize=xlabel_fsize)
            ax.set_ylabel(" ".join(ylabel.upper()), fontproperties=fm, fontsize=ylabel_fsize)
            if is_3d:
                ax.set_zlabel(" ".join(zlabel.upper()), fontproperties=fm, fontsize=11)
    
    if is_image:
        if not tufte_style:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#cccccc')
                spine.set_linewidth(0.5)
                
        fig = ax.get_figure()
        if fig:
            fig.tight_layout(pad=1.0 if tufte_style else 0.5)
        return
    
    # Format axis ticks for non-image plots
    format_axis_ticks(ax, 'x', sharex, is_3d, fm, xtick_fsize)
    format_axis_ticks(ax, 'y', sharey, is_3d, fm, ytick_fsize)
    if is_3d:
        format_axis_ticks(ax, 'z', False, is_3d, fm, 11)
    
    # Format spines
    ylabels = [label.get_text() for label in ax.get_yticklabels()]
    ylabels_fmt = [format_tick_label(label) for label in ylabels] if all(ylabels) else None
    format_spines(ax, ylabels_fmt, sharey, timestamp, tufte_style)
    
    if tufte_style and range_frame and not is_3d:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_bounds(x_min, x_max)
        ax.spines['left'].set_bounds(y_min, y_max)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
    
    if ctab and "patches" in str(ax.get_children()):
        handles, labels = ax.get_legend_handles_labels()
        n_plots = len(labels) if labels else 1
        n_bars = sum(1 for x in ax.get_children() if "Rectangle" in str(x)) - 1
        len_elems = n_bars // n_plots if n_plots > 0 else n_bars
        
        if len_elems > 0:
            hatches = [p for p in PATTERNS for _ in range(len_elems)]
            for bar, hatch in zip(ax.patches, hatches):
                bar.set_hatch(hatch)
    
    existing_legend = ax.get_legend()
    
    if legend:
        if existing_legend is not None:
            existing_legend.set_title(legd_title.upper() if not tufte_style else legd_title)
            for text in existing_legend.get_texts():
                text.set_fontproperties(fm)
                text.set_fontsize(11)
                if not tufte_style:
                    text.set_text(text.get_text().upper())
            existing_legend.get_title().set_fontproperties(fm)
            if tufte_style:
                existing_legend.get_frame().set_facecolor('white')
                existing_legend.get_frame().set_linewidth(0)
        else:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                if tufte_style:
                    legend_labels = legd_labels if legd_labels else labels
                    _legend = ax.legend(handles, legend_labels, title=legd_title,
                                    prop=fm, fontsize=11, loc=legd_loc, frameon=False)
                else:
                    legend_labels = legd_labels if legd_labels else [x.upper() for x in labels]
                    _legend = ax.legend(handles, legend_labels, title=legd_title.upper(),
                                    prop=fm, fontsize=11, loc=legd_loc)
                _legend.get_title().set_fontproperties(fm)
    
    if grid:
        if tufte_style:
            ax.grid(color="#E0E0E0", linestyle=":", linewidth=0.3, alpha=0.9, which='major')
            ax.minorticks_off()
        else:
            ax.grid(color="gray", linestyle="--", linewidth=0.5)
            
    for text_obj in ax.texts:
        text_obj.set_fontproperties(fm)
        if not tufte_style and hasattr(text_obj, 'get_text'):
            try:
                text = text_obj.get_text()
                if not text.replace(".", "").replace("-", "").isdigit():
                    text_obj.set_text(text.upper())
            except (AttributeError, TypeError):
                pass
    
    figure = ax.get_figure()
    for child in figure.get_children():
        if hasattr(child, 'ax') and hasattr(child, 'set_label'):
            if hasattr(child, 'ax'):
                colorbar_ax = child.ax
                if hasattr(colorbar_ax, 'get_ylabel') and colorbar_ax.get_ylabel():
                    colorbar_ax.set_ylabel(
                        colorbar_ax.get_ylabel().upper() if not tufte_style else colorbar_ax.get_ylabel(),
                        fontproperties=fm
                    )
                for label in colorbar_ax.get_yticklabels():
                    label.set_fontproperties(fm)
    
    for ax_obj in figure.get_axes():
        if hasattr(ax_obj, 'title') and ax_obj.title is not None:
            ax_obj.title.set_fontproperties(fm)
            ax_obj.title.set_fontsize(title_fsize)
            if not tufte_style:
                ax_obj.title.set_text(ax_obj.title.get_text().upper())
