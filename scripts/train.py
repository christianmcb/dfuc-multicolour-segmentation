"""Train a U-Net with a chosen colour-channel set.

    python scripts/train.py --config configs/hsv_lab.yaml --data_root /path/to/data

Settings come from the YAML file; any command-line flag overrides it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dfuc_colour.training import main  # noqa: E402

if __name__ == '__main__':
    main()
