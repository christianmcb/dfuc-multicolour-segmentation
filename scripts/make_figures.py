"""Plot validation curves from the saved training logs.

    python scripts/make_figures.py

Writes results/figures/val_curves.png. Uses only results/runs/*/log.csv; the paper's qualitative
figures (predictions) cannot be regenerated because no predictions were saved.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RUNS = [('baseline_bs8', 'RGB baseline (saved log)'), ('hsv_lab_single_tensor', 'HSV + LAB')]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for run, label in RUNS:
        df = pd.read_csv(os.path.join(ROOT, 'results', 'runs', run, 'log.csv'))
        axes[0].plot(df['epoch'], df['val_iou'], label=label)
        axes[1].plot(df['epoch'], df['val_loss'], label=label)
    axes[0].set_ylabel('validation IoU (batch IoU)')
    axes[1].set_ylabel('validation loss (0.5 BCE + Dice)')
    for ax in axes:
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    out_dir = os.path.join(ROOT, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'val_curves.png'), dpi=150)
    print('wrote', os.path.join(out_dir, 'val_curves.png'))


if __name__ == '__main__':
    main()
