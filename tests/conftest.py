import os

import cv2
import numpy as np
import pytest


@pytest.fixture
def bgr_image():
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, size=(48, 64, 3)).astype(np.uint8)


def write_synthetic_dataset(root, name='synthetic', n_train=8, n_val=4, h=48, w=64, seed=0):
    """Random images with a random rectangular 'wound' mask. Plumbing tests only."""
    rng = np.random.RandomState(seed)
    for split, n in (('train', n_train), ('val', n_val)):
        img_dir = os.path.join(root, name, split, 'images')
        mask_dir = os.path.join(root, name, split, 'masks')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        for i in range(n):
            img = rng.randint(0, 256, size=(h, w, 3)).astype(np.uint8)
            mask = np.zeros((h, w), np.uint8)
            y, x = rng.randint(0, h // 2), rng.randint(0, w // 2)
            mask[y:y + h // 3, x:x + w // 3] = 255
            cv2.imwrite(os.path.join(img_dir, '%s_%d.jpg' % (split, i)), img)
            cv2.imwrite(os.path.join(mask_dir, '%s_%d.png' % (split, i)), mask)
    return os.path.join(root)
