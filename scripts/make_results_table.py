"""Regenerate the results tables from results/ (saved logs and numbers transcribed from the paper).

    python scripts/make_results_table.py

Writes results/runs_summary.csv and results/results_table.md and prints the latter.
Nothing is trained or evaluated here.
"""
import glob
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS = os.path.join(ROOT, 'results')
MODELS = ['RGB', 'RGB + Y', 'RGB + A', 'HSV + LAB']
# Which saved run corresponds to each paper row, if any.
PAPER_TO_RUN = {'RGB': 'baseline_bs8', 'HSV + LAB': 'hsv_lab_single_tensor'}
T2_FIELDS = [('epoch', 'epoch'), ('loss', 'train_loss'), ('iou', 'train_iou'), ('dice_score', 'train_dice'),
             ('val_loss', 'val_loss'), ('val_iou', 'val_iou'), ('val_dice_score', 'val_dice')]


def best_epoch_rows():
    rows = []
    for log_path in sorted(glob.glob(os.path.join(RESULTS, 'runs', '*', 'log.csv'))):
        run = os.path.basename(os.path.dirname(log_path))
        df = pd.read_csv(log_path)
        i = df['val_iou'].idxmax()   # train.py saves the checkpoint at the highest val_iou
        r = df.loc[i]
        rows.append(dict(run=run, epochs_run=len(df), best_epoch=int(r['epoch']),
                         epoch=int(r['epoch']), loss=r['loss'], iou=r['iou'],
                         dice_score=r.get('dice_score', float('nan')), val_loss=r['val_loss'],
                         val_iou=r['val_iou'], val_dice_score=r.get('val_dice_score', float('nan'))))
    return pd.DataFrame(rows)


def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(out)


def main():
    summary = best_epoch_rows()
    summary.drop(columns='epoch').round(4).to_csv(os.path.join(RESULTS, 'runs_summary.csv'), index=False)
    saved = summary.set_index('run')
    paper = pd.read_csv(os.path.join(RESULTS, 'paper_reported.csv'))

    # --- Table 2: validation, with a check against the saved log
    t2 = paper[paper.table == 'table2'].pivot(index='model', columns='metric', values='value').loc[MODELS]
    lines = []
    check_rows = []
    for m in MODELS:
        p = t2.loc[m]
        run = PAPER_TO_RUN.get(m)
        if run is None:
            status = 'no saved log'
        else:
            s = saved.loc[run]
            same = all(round(float(s[a]), 4) == round(float(p[b]), 4) for a, b in T2_FIELDS)
            status = ('matches ' if same else 'differs from ') + run
            if not same:
                status += ' (saved: epoch %d, val IoU %.4f, val Dice %.4f)' % (s['best_epoch'], s['val_iou'], s['val_dice_score'])
        check_rows.append({'Colour space': m, 'Epoch': int(p['epoch']), 'T-Loss': p['train_loss'], 'T-IoU': p['train_iou'],
                           'T-Dice': p['train_dice'], 'V-Loss': p['val_loss'], 'V-IoU': p['val_iou'], 'V-Dice': p['val_dice'],
                           'Saved-log check': status})
    lines += ['## Validation (paper Table 2), best epoch by validation IoU, one run per row', '',
              md(pd.DataFrame(check_rows)), '']

    # --- Table 3: test sets, plus differences from the RGB row computed here
    for tab, title in (('table3', 'Test sets (paper Table 3)'), ('table4', 'DFUC2022 test by wound size (paper Table 4)')):
        lines += ['## ' + title + ', no saved output in this repository', '']
        sub = paper[paper.table == tab]
        for group in sub.group.unique():
            g = sub[sub.group == group].pivot(index='model', columns='metric', values='value').loc[MODELS]
            g = g[['fne', 'fpe', 'iou', 'dice']]
            d = (g - g.loc['RGB']).round(4)
            out = pd.DataFrame({'Colour space': MODELS})
            for c in g.columns:
                out[{'fne': 'FNE', 'fpe': 'FPE', 'iou': 'IoU', 'dice': 'Dice'}[c]] = g[c].values
            out['dIoU vs RGB'] = d['iou'].map(lambda x: '%+.4f' % x).values
            out['dDice vs RGB'] = d['dice'].map(lambda x: '%+.4f' % x).values
            lines += ['**' + group + '**', '', md(out.round(4)), '']
    lines += ['dIoU / dDice are differences between paper values, computed by this script.', '']

    # --- Saved-log summary of the runs the paper discusses
    key = ['baseline_bs2', 'baseline_bs4', 'baseline_bs8', 'baseline_bs16', 'hsv_lab_single_tensor', 'rgb_a_single_tensor',
           'rgb_log_a', 'rgb_luminance_1', 'rgb_luminance_2', 'rgb_luminance_3', 'rgb_luminance_4', 'rgb_luminance_5']
    ks = summary[summary.run.isin(key)].set_index('run').loc[[k for k in key if k in saved.index]].reset_index()
    ks = ks[['run', 'epochs_run', 'best_epoch', 'val_loss', 'val_iou', 'val_dice_score']].round(4)
    lines += ['## Saved logs of selected runs (best epoch by validation IoU)', '', md(ks), '']

    text = '\n'.join(lines)
    with open(os.path.join(RESULTS, 'results_table.md'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
