"""Colour-channel merging ("tensor merging") applied to a BGR uint8 image as read by cv2.imread.

Each function takes an H x W x 3 uint8 BGR array and returns an H x W x C uint8 array.
The arithmetic is taken unchanged from the original dataset classes in
archive/original_datasets/ (see the docstrings for the source file and lines).
Scaling to [0, 1] happens afterwards, in dataset.Dataset.
"""
from functools import partial

import cv2
import numpy as np


def rgb(img):
    """Baseline: the three channels as read (BGR order). 3 channels."""
    return img


def rgb_plus_luminance(img, power=5):
    """B, G, R + luminance. 4 channels.

    Source: dataset_RGB_L.py:69-74. Luminance uses BT.709 weights on BGR order, is divided
    by its per-image maximum, raised to `power`, divided by its per-image maximum again and
    scaled to uint8. The source file holds power=5; the paper's "RGB+Y" exponent is not
    recorded (see configs/rgb_y.yaml).
    """
    i1, i2, i3 = cv2.split(img)
    l = (i1*0.0722 + i2*0.7152 + i3*0.2126)
    l = np.array(((l) / np.max(l)) * 255, dtype='float32')

    luminance = np.array((l**power / np.max(l**power)) * 255, dtype='uint8')

    return cv2.merge((i1, i2, i3, luminance))


def rgb_plus_log_a(img):
    """B, G, R + log-transformed CIELAB a-channel. 4 channels.

    Source: dataset_RGB_A.py:66-72. OpenCV 8-bit LAB, a -> log(a + 1e-5), negatives set to 0,
    divided by the per-image maximum, scaled to uint8. (The unused HSV conversion in the
    original was dropped; it did not affect the output.)
    """
    i1, i2, i3 = cv2.split(img)
    imageLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    k1, k2, k3 = cv2.split(imageLAB)
    k2 = np.log(k2+0.00001)
    k2 = np.where(k2 < 0, 0, k2)
    k2 = np.array((k2 / np.max(k2)) * 255, dtype='uint8')

    return cv2.merge((i1, i2, i3, k2))


def hsv_plus_lab(img):
    """H, S, V, L, a, b (OpenCV 8-bit conversions). 6 channels, no RGB.

    Source: dataset_RGB_HSV.py:59-66. Hue is 0-179 in OpenCV 8-bit, so after /255 it
    reaches about 0.70.
    """
    imageHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imageLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    j1, j2, j3 = cv2.split(imageHSV)
    k1, k2, k3 = cv2.split(imageLAB)
    return cv2.merge((j1, j2, j3, k1, k2, k3))


# name -> (function, number of output channels)
COLOUR_TRANSFORMS = {
    'rgb': (rgb, 3),
    'rgb_y': (rgb_plus_luminance, 4),
    'rgb_a': (rgb_plus_log_a, 4),
    'hsv_lab': (hsv_plus_lab, 6),
}


def get_colour_transform(name, **kwargs):
    """Return (callable, n_channels) for a transform name; kwargs are bound (e.g. power=4)."""
    if name not in COLOUR_TRANSFORMS:
        raise ValueError('unknown colour transform %r; choose from %s' % (name, sorted(COLOUR_TRANSFORMS)))
    fn, n_channels = COLOUR_TRANSFORMS[name]
    return (partial(fn, **kwargs) if kwargs else fn), n_channels
