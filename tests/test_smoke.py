"""Plumbing check on a tiny synthetic dataset: random images and masks, 2 epochs, CPU.
It shows that training, checkpointing, logging and evaluation run end to end.
It says nothing about the quality of the results."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from conftest import write_synthetic_dataset
from dfuc_colour.training import main

ROOT = os.path.join(os.path.dirname(__file__), '..')


@pytest.mark.parametrize('colour,channels', [('rgb', 3), ('rgb_y', 4), ('rgb_a', 4), ('hsv_lab', 6)])
def test_train_and_evaluate_on_synthetic_data(tmp_path, colour, channels):
    data_root = write_synthetic_dataset(str(tmp_path / 'data'))
    out_dir = str(tmp_path / 'runs')
    main(['--dataset', 'synthetic', '--data_root', data_root, '--output_dir', out_dir, '--name', 'smoke',
          '--colour_transform', colour, '--epochs', '2', '--batch_size', '2', '--num_workers', '0'])

    run = os.path.join(out_dir, 'smoke')
    log = pd.read_csv(os.path.join(run, 'log.csv'))
    assert len(log) == 2
    assert list(log.columns) == ['epoch', 'lr', 'loss', 'iou', 'dice_score', 'val_loss', 'val_iou', 'val_dice_score']
    assert log.notna().all().all()
    assert os.path.exists(os.path.join(run, 'model.pth'))
    assert yaml.safe_load(open(os.path.join(run, 'config.yml')))['input_channels'] == channels

    result = str(tmp_path / 'result.json')
    subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'evaluate.py'), '--run', run,
                    '--data_root', data_root, '--split', 'val', '--num_workers', '0', '--out', result],
                   check=True, capture_output=True)
    metrics = json.load(open(result))
    assert 0.0 <= metrics['iou'] <= 1.0 and 0.0 <= metrics['val_dice_score'] <= 1.0


def test_yaml_config_is_read_and_cli_overrides_it(tmp_path):
    from dfuc_colour.training import parse_args
    cfg = tmp_path / 'c.yaml'
    cfg.write_text('colour_transform: hsv_lab\nbatch_size: 4\n')
    c = parse_args(['--config', str(cfg), '--batch_size', '2'])
    assert c.colour_transform == 'hsv_lab' and c.input_channels == 6 and c.batch_size == 2


def test_repository_configs_are_consistent():
    from dfuc_colour.training import parse_args
    for name, ch in [('baseline_rgb', 3), ('rgb_y', 4), ('rgb_a', 4), ('hsv_lab', 6)]:
        c = parse_args(['--config', os.path.join(ROOT, 'configs', name + '.yaml')])
        assert c.input_channels == ch
        assert (c.batch_size, c.early_stopping, c.lr, c.patience, c.loss) == (8, 20, 0.001, 6, 'BCEDiceLoss')
