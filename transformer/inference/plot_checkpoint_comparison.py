import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

import utils.file_management as fm
#so that I can import from a directory two levels up
from evaluation.graph_helper import calc_bpos_behavior
from utils.parse_data import (align_predictions_with_gt, get_data_filenames,
                              parse_simulated_data)

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../behavior-helpers/')))
from bh.visualization import plot_trials as pts


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
    color_map_copy = color_map.copy()  # Create a copy so we don't modify the original
    
    if 'ground truth' in color_map_copy:
        del color_map_copy['ground truth']  # Remove ground truth from colormap

    # Get checkpoint numbers and sort them
    checkpoint_labels = list(color_map_copy.keys())
    checkpoint_numbers = [int(label.split("_cp")[-1].replace(".txt", "")) 
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
    
    # Create the normalization
    norm = BoundaryNorm(bounds, discrete_cmap.N)
    
    # Create the scalar mappable
    sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])
    
    # Add colorbar with specific ticks at checkpoint numbers
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), 
                       ticks=checkpoint_numbers,
                       **colorbar_kwargs)
    cbar.set_label('Checkpoint')
    
    # Remove any existing legends if requested
    if remove_legends:
        for ax in axs.ravel():
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


def main(run=None, suffix: str = 'v'):
    
    """Plot behavior comparisons across checkpoints and domains."""
    sns.set_theme(style='ticks', font_scale=1.0, rc={'axes.labelsize': 12,
                  'axes.titlesize': 12, 'savefig.transparent': False})

    import glob

    # Files will automatically use latest run if run=None
    run = run or fm.get_latest_run()
    # Find all checkpoint files
    checkpoint_files = sorted(glob.glob(os.path.join(fm.get_run_dir(run), 'seqs', "pred_model*cp*.txt")), key=lambda x: int(x.split('_cp')[-1].replace('.txt', '')))
    indices_files = sorted(glob.glob(os.path.join(fm.get_run_dir(run), 'seqs', "pred_indices_model*cp*.txt")), key=lambda x: int(x.split('_cp')[-1].replace('.txt', '')))
    model_files = glob.glob(os.path.join(fm.get_run_dir(run), 'models', "model_*.pth"))

    if not checkpoint_files:
        print(f"No checkpoint models found in run {run}")
        return
    elif len(checkpoint_files) == 1:
        print(f"Only one checkpoint model found in run {run}")
        return
    model_name = model_files[0].split('/')[-1].split('_cp')[0]

    # Load ground truth data once
    files = get_data_filenames(run, suffix=suffix)
    gt_events = parse_simulated_data(*files)
    domains = sorted(gt_events['domain'].unique())

    gt_policies = pts.calc_conditional_probs(gt_events, htrials=2, sortby='pevent', pred_col='switch', add_grps='domain')
    gt_policies['model'] = 'ground truth'
    # Create figure with two subplots for each domain
    fig, axes = plt.subplots(3, len(domains), figsize=(4.5*len(domains), 6),
                             sharex=False, layout='constrained')
    
    # Convert axes to numpy array for consistent handling regardless of domain count
    axes = np.array(axes)
    
    # If we have only one domain, reshape to maintain 2D structure
    if len(domains) == 1:
        axes = axes.reshape(3, 1)

    colors = sns.color_palette('viridis', n_colors=len(checkpoint_files))
    cmap = {'ground truth': 'k'}

    for pred_file, indices_file, color in zip(checkpoint_files, indices_files, colors):
        
        # Extract checkpoint numbers
        assert pred_file.split('_cp')[-1].replace('.txt', '') == indices_file.split('_cp')[-1].replace('.txt', ''), (
            f"Checkpoint numbers don't match between prediction file ({pred_file}) and indices file ({indices_file})"
        )
        
        # Load predictions for this checkpoint
        predictions = fm.read_sequence(pred_file)
        with open(indices_file, 'r') as f:
            indices = [int(line.strip()) for line in f]
        events = align_predictions_with_gt(gt_events, predictions, indices)
        bpos_ = calc_bpos_behavior(events,
                                   add_cond_cols=['domain', 'session'],
                                   add_agg_cols=['pred_switch', 'pred_selected_high'])

        label = pred_file.split("_cp")[-1].replace(".txt", "")
        for ax_, (domain, bpos_domain) in zip(axes.T, bpos_.groupby('domain')):
            sns.lineplot(bpos_domain.query('iInBlock.between(-11, 21)'),
                         x='iInBlock', y='pred_selected_high', ax=ax_[0], color=color, legend=False)
            sns.lineplot(bpos_domain.query('iInBlock.between(-11, 21)'),
                         x='iInBlock', y='pred_switch', ax=ax_[1], color=color, label=label)
            
        pred_policies = pts.calc_conditional_probs(events, htrials=2, sortby='pevent', pred_col='pred_switch', add_grps='domain')
        pred_policies['model'] = label
        gt_policies = pd.concat([gt_policies, pred_policies])
        cmap[label] = color
    # Ground truth data -- mimic as a checkpoint
    for ax_, (domain, bpos_domain) in zip(axes.T, bpos_.groupby('domain')):
        sns.lineplot(bpos_domain.query('iInBlock.between(-11, 21)'),
                     x='iInBlock', y='selHigh', ax=ax_[0], color='k', legend=False, linewidth=3, errorbar=None)
        sns.lineplot(bpos_domain.query('iInBlock.between(-11, 21)'),
                     x='iInBlock', y='Switch', ax=ax_[1], color='k', label='ground truth', linewidth=3, errorbar=None)
        ax_[0].vlines(x=0, ymin=-1, ymax=1.5, ls='--', color='k', zorder=0)
        ax_[1].vlines(x=0, ymin=-1, ymax=1.5, ls='--', color='k', zorder=0)
        ax_[0].set(title=domain, xlim=(-10, 20), ylabel='P(high)', ylim=(0, 1.1))
        ax_[1].set(xlabel='block position', xlim=(-10, 20),
                   ylabel='P(switch)', ylim=(0, 0.45))

        print(gt_policies)
        _, ax_[2] = pts.plot_sequences(gt_policies.query('model == "ground truth" & domain == @domain'), ax=ax_[2])
        fig, ax_[2] = pts.plot_sequence_points(gt_policies.query('model != "ground truth" & domain == @domain'), grp='model', palette=cmap, yval='pevent', size=3, ax=ax_[2], fig=fig, legend=False)

    # Modify legend display based on domains
    # if len(domains) > 1:
    #     [ax_[1].get_legend().set_visible(False) for ax_ in axes.T[:-1]]
    #     axes.T[-1][1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Checkpoint')
    #     # add_checkpoint_colorbar(fig, axes, checkpoint_files)
    # else:
        # axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Checkpoint')
    add_checkpoint_colorbar(fig, axes, cmap)
    
    # fig.subplots_adjust(right=0.85)  # Makes room for the legend on the right

    sns.despine()
    # curr_dir = os.path.dirname(os.path.abspath(__file__))
    # new_dir = os.path.join(curr_dir, '..', 'test')
    # fig_path = os.path.join(new_dir, f'bpos_checkpoints_{model_name}.png')
    fig_path = fm.get_experiment_file(f'bpos_checkpoints_{model_name}.png', run, subdir='predictions')
    fig.savefig(fig_path, bbox_inches='tight')
    print(f'Saved checkpoint comparison plot to {fig_path}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    args = parser.parse_args()
    main(run=args.run)
