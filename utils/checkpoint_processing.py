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

def add_checkpoint_colorbar(fig, axs, color_map, ground_truth=True,
                           remove_legends=True, colorbar_kwargs=None):
    """
    Add a colorbar for checkpoint numbers instead of a legend.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to add the colorbar to
    axes : numpy.ndarray
        Array of axes objects
    color_map : dict
        Dictionary mapping model names to colors
    ground_truth : bool, default=True
        Whether to add a separate legend for ground truth
    remove_legends : bool, default=True
        Whether to remove existing legends
    colorbar_kwargs : dict, optional
        Additional kwargs to pass to fig.colorbar()
        
    Returns:
    --------
    cbar : matplotlib.colorbar.Colorbar
        The colorbar object
    """
    # Default colorbar kwargs
    if colorbar_kwargs is None:
        colorbar_kwargs = {'shrink': 0.3, 'pad': 0.02, 'location': 'right'}

    # Extract checkpoint numbers as integers for colorbar
    color_map_copy = {k: v for k, v in color_map.copy().items() if 'cp' in k}
    if not color_map_copy:
        color_map_copy = {k: v for k, v in color_map.copy().items() if 'ground truth' not in k}

    # Get checkpoint numbers and sort them
    checkpoint_labels = list(color_map_copy.keys())
    checkpoint_numbers = [int(label.split("cp")[-1].replace(".txt", "")) 
                          for label in checkpoint_labels]
    checkpoint_numbers.sort()  # Sort in ascending order

    # Create discrete boundaries and select colors for each checkpoint
    from matplotlib.colors import BoundaryNorm

    # Create boundaries slightly below and above each checkpoint number
    bounds = np.array([num - 0.5 for num in checkpoint_numbers] + 
                      [checkpoint_numbers[-1] + 0.5])

    # Get N evenly spaced colors from viridis
    cmap = plt.cm.viridis
    n_colors = len(checkpoint_numbers)
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    
    # Create a ListedColormap with these colors
    from matplotlib.colors import ListedColormap
    discrete_cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, discrete_cmap.N)
    sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])
    
    # Add colorbar with specific ticks at checkpoint numbers
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist() if isinstance(axs, np.ndarray) else axs, 
                       ticks=checkpoint_numbers,
                       **colorbar_kwargs)
    cbar.set_label('Checkpoint')

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