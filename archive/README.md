# Archive

Code from the original MSc working directory that the refactored package in `src/` does not use.
Nothing here is maintained or tested.

- `original_datasets/` and `original_training/`: the files the refactor was derived from, kept so that
  every function in `src/` can be compared line by line with the original. `dataset_RGB_L.py` and
  `dataset_RGB_A.py` hold the *last* edited variant (luminance^5 and log(a)), not necessarily the one
  used for each paper row. `dataset_HSV_LAB.py`, imported by `train_HSV_LAB.py`, was not in the directory.
- `augmentation/`: offline scripts that wrote augmented copies of the training set for the augmentation
  experiments (not in the short paper). `sharpen.py` and `mask_dilation.py` have each other's contents.
- `legacy/`: earlier training script, a broken TensorFlow FCN baseline, and the full `archs.py` /
  `losses.py` including the unused NestedUNet, FocalLoss and LovaszHingeLoss.

The only edit made to these files is replacing absolute home-directory paths with `<DATA_ROOT>`.
