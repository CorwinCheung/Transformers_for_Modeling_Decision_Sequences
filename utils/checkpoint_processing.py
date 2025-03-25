import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.append(os.path.abspath(os.path.join(__file__, '../')))
import utils.file_management as fm
from utils.parse_data import load_trained_model


def get_checkpoint_files(run, include_final=True, subdir='models', pattern='model_', ext='.pth'):
    """Get all checkpoint model files for a run, sorted by checkpoint number."""
    model_files = glob.glob(os.path.join(fm.get_run_dir(run), subdir, f"{pattern}*cp*{ext}"))

    # Sort by checkpoint number
    model_files = sorted(model_files, 
                         key=lambda x: int(x.split('_cp')[-1].replace(ext, '')))

    # Add the final model if it exists and is requested
    if include_final:
        final_model = glob.glob(os.path.join(fm.get_run_dir(run), subdir, f"{pattern}*{ext}"))
        final_model = [m for m in final_model if '_cp' not in m]
        if final_model:
            model_files.append(final_model[0])

    return model_files


def process_checkpoints(run, processor_fn, reference_type='final', include_final=True, 
                        save_results=False, processor_kwargs=None):
    """
    Generic framework for processing model checkpoints.

    Parameters:
    -----------
    run : int
        Run number to analyze
    processor_fn : callable
        Function that processes each model, with signature:
        processor_fn(model, checkpoint_num, is_reference, model_info, **kwargs) -> result
    reference_type : str, default='final'
        Which model to use as reference ('final', 'first', or None)
    include_final : bool, default=True
        Whether to include the final model in the analysis
    save_results : bool, default=False
        Whether to save intermediate results
    processor_kwargs : dict, optional
        Additional keyword arguments to pass to the processor function

    Returns:
    --------
    dict
        Contains processed results, checkpoint numbers, model labels, reference index,
        and any other data returned by the processor function
    """
    device = torch.device('cpu')  # Use CPU for consistency

    # Get all checkpoint models
    model_files = get_checkpoint_files(run, include_final=include_final)

    if not model_files:
        print(f"No models found for run {run}")
        return None

    print(f"Found {len(model_files)} models for run {run}")

    # Determine the reference model
    if reference_type == 'final' and include_final:
        reference_idx = -1  # Use the last model
    elif reference_type == 'first':
        reference_idx = 0   # Use the first model
    else:
        reference_idx = None

    # Process each model
    all_results = []
    checkpoint_numbers = []
    model_labels = []    
    # Ensure processor_kwargs is a dictionary
    if processor_kwargs is None:
        processor_kwargs = {}

    for i, model_file in enumerate(model_files):
        # Extract checkpoint number and label
        if '_cp' in model_file:
            cp_num = int(model_file.split('_cp')[-1].replace('.pth', ''))
            label = f"cp{cp_num}"
        else:
            cp_num = float('inf')  # Final model
            label = "final"
    
        model_name = os.path.basename(model_file).replace('.pth', '')
        model, model_info, config = load_trained_model(run, model_name=model_name, device=device, weights_only=True)

        # Determine if this is the reference model
        is_reference = (i == reference_idx) if reference_idx is not None else False

        # Process the model
        result = processor_fn(
            model=model,
            checkpoint_num=cp_num,
            is_reference=is_reference,
            model_info=model_info,
            config=config,
            **processor_kwargs
        )

        all_results.append(result)
        checkpoint_numbers.append(cp_num)
        model_labels.append(label)

    # Return all the data
    return {
        'results': all_results,
        'checkpoint_numbers': checkpoint_numbers,
        'model_labels': model_labels,
        'reference_idx': reference_idx,
        'run': run
    }

def generate_checkpoint_colormap(checkpoint_labels=None, checkpoint_numbers=None, cmap_name='viridis'):
    """
    Generate a colormap for checkpoints that can be used across different plotting functions.
    
    Parameters:
    -----------
    checkpoint_labels : list, optional
        List of checkpoint labels (e.g., ['cp10', 'cp20', ...])
    checkpoint_numbers : list, optional
        List of checkpoint numbers (e.g., [10, 20, ...])
    cmap_name : str, default='viridis'
        Name of the matplotlib colormap to use
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'colors': dict mapping checkpoint labels to colors
        - 'cmap': the matplotlib colormap
        - 'norm': the normalization
        - 'is_log_spaced': boolean indicating if checkpoints are log-spaced
        - 'checkpoint_numbers': sorted list of checkpoint numbers
    """
    from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
    
    # Extract checkpoint numbers if not provided directly
    if checkpoint_numbers is None and checkpoint_labels is not None:
        checkpoint_numbers = [int(label.split("cp")[-1].replace(".txt", "")) 
                             for label in checkpoint_labels if 'cp' in label]
    
    if not checkpoint_numbers:
        return None
        
    checkpoint_numbers = sorted(checkpoint_numbers)
    
    # Determine if checkpoints are log-spaced
    is_log_spaced = False
    if len(checkpoint_numbers) > 3:
        intervals = [checkpoint_numbers[i+1] - checkpoint_numbers[i] for i in range(len(checkpoint_numbers)-1)]
        ratios = [intervals[i+1] / intervals[i] if intervals[i] > 0 else 1 for i in range(len(intervals)-1)]
        
        if sum(ratios) / len(ratios) > 1.3:
            is_log_spaced = True
    
    # Create base colormap
    cmap = plt.cm.get_cmap(cmap_name)
    
    # Setup normalization and colormap based on checkpoint spacing
    if is_log_spaced:
        # For log-spaced checkpoints, use continuous mapping
        norm = Normalize(vmin=checkpoint_numbers[0], vmax=checkpoint_numbers[-1])
        colors = {f"{num}": cmap(norm(num)) for num in checkpoint_numbers}
    else:
        # For linearly-spaced checkpoints, use discrete colors
        n_colors = len(checkpoint_numbers)
        discrete_colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
        
        bounds = np.array([num - 0.5 for num in checkpoint_numbers] + 
                         [checkpoint_numbers[-1] + 0.5])
        discrete_cmap = ListedColormap(discrete_colors)
        norm = BoundaryNorm(bounds, discrete_cmap.N)
        
        colors = {f"cp{num}": discrete_colors[i] for i, num in enumerate(checkpoint_numbers)}
        cmap = discrete_cmap
    
    # Add final model if needed
    if checkpoint_labels and 'final' in checkpoint_labels:
        colors['final'] = 'red'  # Special color for final model
    
    # Return complete colormap package
    return {
        'colors': colors,
        'cmap': cmap,
        'norm': norm,
        'is_log_spaced': is_log_spaced,
        'checkpoint_numbers': checkpoint_numbers
    }

def add_checkpoint_colorbar(fig, axs, color_map, ground_truth=True,
                            remove_legends=True, colorbar_kwargs=None, max_labels=8):
    """
    Add a colorbar for checkpoint numbers instead of a legend.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to add the colorbar to
    axes : numpy.ndarray
        Array of axes objects
    color_map : dict
        Dictionary mapping model names to colors, or the output from generate_checkpoint_colormap
    ground_truth : bool, default=True
        Whether to add a separate legend for ground truth
    remove_legends : bool, default=True
        Whether to remove existing legends
    colorbar_kwargs : dict, optional
        Additional kwargs to pass to fig.colorbar()
    max_labels : int, default=8
        Maximum number of checkpoint labels to show on the colorbar
        
    Returns:
    --------
    cbar : matplotlib.colorbar.Colorbar
        The colorbar object
    """
    # Default colorbar kwargs
    if colorbar_kwargs is None:
        colorbar_kwargs = {'shrink': 0.3, 'pad': 0.02, 'location': 'right'}

    # Check if color_map is already processed or just a simple dict
    if isinstance(color_map, dict) and any(key in color_map for key in ['colors', 'cmap', 'norm']):
        # Already processed map
        colormap_data = color_map
        checkpoint_numbers = colormap_data['checkpoint_numbers']
        is_log_spaced = colormap_data['is_log_spaced']
        cmap = colormap_data['cmap']
        norm = colormap_data['norm']
    else:
        # Extract checkpoint labels/numbers and process
        color_map_copy = {k: v for k, v in color_map.copy().items() if 'cp' in k}
        if not color_map_copy:
            color_map_copy = {k: v for k, v in color_map.copy().items() if 'ground truth' not in k}

        checkpoint_labels = list(color_map_copy.keys())
        colormap_data = generate_checkpoint_colormap(checkpoint_labels)
        
        if colormap_data is None:
            return None
            
        checkpoint_numbers = colormap_data['checkpoint_numbers']
        is_log_spaced = colormap_data['is_log_spaced']
        cmap = colormap_data['cmap']
        norm = colormap_data['norm']
    
    # Create scalar mappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Add colorbar to figure
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist() if isinstance(axs, np.ndarray) else axs,
                       ticks=checkpoint_numbers, **colorbar_kwargs)
    cbar.set_label('Checkpoint')
    
    # Handle label frequency reduction for readability
    if len(checkpoint_numbers) > max_labels:
        n = len(checkpoint_numbers)
        
        # Always include first and last checkpoints
        if n <= 2:
            indices = list(range(n))
        else:
            # For logarithmic spacing, use logarithmic distribution of labels
            if is_log_spaced:
                log_min = np.log10(1 + checkpoint_numbers[0]) if checkpoint_numbers[0] >= 0 else 0
                log_max = np.log10(1 + checkpoint_numbers[-1])
                log_steps = np.linspace(log_min, log_max, max_labels)
                
                # Convert log steps back to checkpoint values
                step_values = [10**x - 1 for x in log_steps]
                
                # Find nearest checkpoint to each log step
                indices = [0]  # Always include first
                for val in step_values[1:-1]:  # Skip first and last
                    idx = min(range(1, n-1), key=lambda i: abs(checkpoint_numbers[i] - val))
                    if idx not in indices:
                        indices.append(idx)
                indices.append(n-1)  # Always include last
            else:
                # For linear spacing, use linear distribution
                step = max(1, (n-2) // (max_labels-2))
                middle_indices = list(range(1, n-1, step))
                middle_indices = middle_indices[:max_labels-2]  # Limit to max_labels - 2
                indices = [0] + middle_indices + [n-1]

        cbar.ax.set_yticklabels([''] * len(checkpoint_numbers))

        for idx in indices:
            cbar.ax.get_yticklabels()[idx].set_text(str(checkpoint_numbers[idx]))
            cbar.ax.get_yticklabels()[idx].set_visible(True)

    # Remove any existing legends if requested
    if remove_legends:
        for ax in axs.ravel() if isinstance(axs, np.ndarray) else [axs]:
            legend = ax.get_legend()
            if legend is not None:
                legend.set_visible(False)

    # Add a separate legend for ground truth if needed
    if ground_truth:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='k', lw=3, label='Ground Truth')]

        fig.legend(handles=legend_elements, 
                  loc='upper right', 
                  bbox_to_anchor=(1.0, 0.35),
                  frameon=False,
                  fontsize='small',
                  title_fontsize='small',
                  borderpad=0.3,
                  handlelength=1.5)
    
    return cbar