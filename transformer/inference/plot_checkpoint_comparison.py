import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

import utils.file_management as fm
#so that I can import from a directory two levels up
from evaluation.graph_helper import calc_bpos_behavior
from utils.parse_data import align_predictions_with_gt, parse_simulated_data


def main(run=None, suffix: str = 'v'):
    
    """Plot behavior comparisons across checkpoints and contexts."""
    sns.set_theme(style='ticks', font_scale=1.0, rc={'axes.labelsize': 12,
                  'axes.titlesize': 12, 'savefig.transparent': False})

    import glob

    # Files will automatically use latest run if run=None
    run = run or fm.get_latest_run()
    # Find all checkpoint files
    checkpoint_files = sorted(glob.glob(os.path.join(fm.get_run_dir(run), 'seqs', "pred_*cp*.txt")))
    model_files = glob.glob(os.path.join(fm.get_run_dir(run), 'models', "model_*.pth"))
    if not checkpoint_files:
        print(f"No checkpoint models found in run {run}")
        return
    model_name = model_files[0].split('/')[-1].split('_cp')[0]

    # Load ground truth data once
    behavior_file = fm.get_experiment_file("behavior_run_{}.txt", run, suffix, subdir='seqs')
    high_port_file = fm.get_experiment_file("high_port_run_{}.txt", run, suffix, subdir='seqs')
    context_file = fm.get_experiment_file("context_transitions_run_{}.txt", run, suffix, subdir='seqs')
    
    gt_events = parse_simulated_data(behavior_file, high_port_file, context_file)
    contexts = sorted(gt_events['context'].unique())

    # Create figure with two subplots for each context
    fig, axes = plt.subplots(2, len(contexts), figsize=(4.5*len(contexts), 6),
                             sharex=True, layout='constrained')
    if len(contexts) == 1:
        axes = [axes]

    colors = sns.color_palette('viridis', n_colors=len(checkpoint_files))
    for pred_file, color in zip(checkpoint_files, colors):
        # Load predictions for this checkpoint
        predictions = fm.read_sequence(pred_file)
        events = align_predictions_with_gt(gt_events, predictions)
        bpos_ = calc_bpos_behavior(events,
                                   add_cond_cols=['context', 'session'],
                                   add_agg_cols=['pred_switch', 'pred_selected_high'])

        label = pred_file.split("_cp")[-1].replace(".txt", "")
        for ax_, (context, bpos_context) in zip(axes.T, bpos_.groupby('context')):
            sns.lineplot(bpos_context.query('iInBlock.between(-11, 21)'),
                         x='iInBlock', y='pred_selected_high', ax=ax_[0], color=color, legend=False)
            sns.lineplot(bpos_context.query('iInBlock.between(-11, 21)'),
                         x='iInBlock', y='pred_switch', ax=ax_[1], color=color, label=label)

    # Ground truth data -- mimic as a checkpoint
    for ax_, (context, bpos_context) in zip(axes.T, bpos_.groupby('context')):
        sns.lineplot(bpos_context.query('iInBlock.between(-11, 21)'),
                     x='iInBlock', y='selHigh', ax=ax_[0], color='k', legend=False, linewidth=3, errorbar=None)
        sns.lineplot(bpos_context.query('iInBlock.between(-11, 21)'),
                     x='iInBlock', y='Switch', ax=ax_[1], color='k', label='ground truth', linewidth=3, errorbar=None)
        ax_[0].vlines(x=0, ymin=-1, ymax=1.5, ls='--', color='k', zorder=0)
        ax_[1].vlines(x=0, ymin=-1, ymax=1.5, ls='--', color='k', zorder=0)
        ax_[0].set(title=context,
                   ylabel='P(pred high)', ylim=(0, 1.1))
        ax_[1].set(xlabel='block position', xlim=(-10, 20),
                   ylabel='P(pred switch)', ylim=(0, 0.3))

    [ax_[1].get_legend().set_visible(False) for ax_ in axes.T[:-1]]
    ax_[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Checkpoint')

    sns.despine()
    fig_path = fm.get_experiment_file(f'bpos_checkpoints_{model_name}.png', run, subdir='predictions')
    fig.savefig(fig_path, bbox_inches='tight')
    print(f'Saved checkpoint comparison plot to {fig_path}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    args = parser.parse_args()
    main(run=args.run)
