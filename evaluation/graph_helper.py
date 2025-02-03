import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import bootstrap

sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))

from utils.file_management import get_experiment_file

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../behavior-helpers/')))
from bh.visualization import plot_trials as pts

sns.set_theme(style='ticks', font_scale=1.0, rc={'axes.labelsize': 12,
              'axes.titlesize': 12, 'savefig.transparent': False})


def calc_bpos_behavior(events, add_cond_cols=['context', 'session'], **kwargs):
    events = events.rename(columns={'block_position': 'iInBlock',
                                    'block_length': 'blockLength',
                                    'selected_high': 'selHigh',
                                    'switch': 'Switch'})
    bpos = pts.calc_bpos_probs(events, add_cond_cols=add_cond_cols, **kwargs)
    return bpos


def plot_bpos_behavior(bpos, run, suffix: str = 'v', errorbar='se',  **kwargs):
    source = kwargs.pop('source', '') + '_'
    fig, axs = pts.plot_bpos_behavior(bpos.query('iInBlock.between(-15, 25)'), errorbar=errorbar, **kwargs)
    [ax.set(xlim=(-10, 20)) for ax in axs]
    axs[0].set(ylim=(0, 1.1))
    axs[1].set(ylim=(0, 0.4))
    bpos_filename = get_experiment_file(f'bpos_{source}{suffix}.png', run)
    fig.savefig(bpos_filename, bbox_inches='tight')
    print(f'saved block position behavior to {bpos_filename}')
    return fig, axs


def plot_conditional_switching(events, seq_length, run, suffix: str = 'v'):

    for context in events.context.unique():
        policies = pts.calc_conditional_probs(
            events.query('context == @context'), htrials=seq_length, sortby='pevent', pred_col='switch')
        fig, ax = pts.plot_sequences(policies)

        if seq_length > 2:
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        if events.context.nunique() == 1:
            fig_path = get_experiment_file(f"cond_switch_{seq_length}{suffix}{context}.png", run)
            fig.savefig(fig_path)
            print(f'saved conditional probabilities for {seq_length} trials to {fig_path}')

    if events.context.nunique() > 1:
        fig, ax, _ = pts.plot_seq_bar_and_points(events, seq_length,
                                        grp='context',
                                        palette='deep',
                                        pred_col='switch')
        fig_path = get_experiment_file(f"cond_switch_{seq_length}{suffix}_context_comparison.png", run)
        fig.savefig(fig_path)
        print(f'saved context comparison for {seq_length} trials to {fig_path}')
