"""Evaluate a saved checkpoint on a split that has masks (normally 'val').

    python scripts/evaluate.py --run runs/hsv_lab --data_root /path/to/data --split val

Uses the same `validate` function, loss and metrics as training, so the numbers are the
validation-style metrics (batch IoU, soft Dice). It is NOT the protocol used for the paper's
test tables (Tables 3 and 4), whose code and predictions are not in this repository.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch  # noqa: E402
import yaml  # noqa: E402

from dfuc_colour import models  # noqa: E402
from dfuc_colour.colourspaces import get_colour_transform  # noqa: E402
from dfuc_colour.training import get_device, make_criterion, make_loader, validate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='run folder containing config.yml and model.pth')
    p.add_argument('--data_root', default=None, help='overrides data_root in the run config')
    p.add_argument('--split', default='val', help='split folder with images/ and masks/')
    p.add_argument('--num_workers', type=int, default=None)
    p.add_argument('--out', default=None, help='optional JSON file for the result')
    args = p.parse_args()

    with open(os.path.join(args.run, 'config.yml')) as f:
        config = yaml.safe_load(f)
    if args.data_root is not None:
        config['data_root'] = args.data_root
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers

    device = get_device()
    kwargs = {'power': config['luminance_power']} if config['colour_transform'] == 'rgb_y' else {}
    colour_transform, _ = get_colour_transform(config['colour_transform'], **kwargs)
    loader = make_loader(config, args.split, colour_transform, shuffle=False, drop_last=False)

    model = models.UNet(num_classes=config['num_classes'], input_channels=config['input_channels'])
    model.load_state_dict(torch.load(os.path.join(args.run, 'model.pth'), map_location=device))
    model = model.to(device)

    result = {k: float(v) for k, v in validate(config, loader, model, make_criterion(config, device), device).items()}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
