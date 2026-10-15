import importlib.util
import os

import cv2
import numpy as np
import pytest

from dfuc_colour import colourspaces as cs

ARCHIVE = os.path.join(os.path.dirname(__file__), '..', 'archive', 'original_datasets')


@pytest.mark.parametrize('name,channels', [('rgb', 3), ('rgb_y', 4), ('rgb_a', 4), ('hsv_lab', 6)])
def test_shape_dtype_and_range(bgr_image, name, channels):
    fn, n = cs.get_colour_transform(name)
    out = fn(bgr_image)
    assert n == channels
    assert out.shape == (48, 64, channels)
    assert out.dtype == np.uint8
    scaled = out.astype('float32') / 255      # what Dataset does
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0


@pytest.mark.parametrize('name', ['rgb_y', 'rgb_a'])
def test_first_three_channels_are_the_original_bgr(bgr_image, name):
    out = cs.get_colour_transform(name)[0](bgr_image)
    assert np.array_equal(out[..., :3], bgr_image)


def test_luminance_channel_spans_to_255_and_is_monotone_in_power(bgr_image):
    # the channel is divided by its own per-image maximum, so the maximum is exactly 255
    for p in (1, 4, 5):
        assert cs.rgb_plus_luminance(bgr_image, power=p)[..., 3].max() == 255
    # a higher power pushes mid values down
    low = cs.rgb_plus_luminance(bgr_image, power=1)[..., 3].astype(float).mean()
    high = cs.rgb_plus_luminance(bgr_image, power=5)[..., 3].astype(float).mean()
    assert high < low


def test_luminance_hand_value():
    # one pure-green pixel and one white pixel (BGR). BT.709 weights on BGR order.
    img = np.array([[[0, 255, 0], [255, 255, 255]]], dtype=np.uint8)
    out = cs.rgb_plus_luminance(img, power=1)[..., 3]
    # luminance: green = 0.7152*255, white = 255 -> after /max*255: white 255, green 0.7152*255 -> 182
    assert out[0, 1] == 255
    assert out[0, 0] == int(np.float32(0.7152 * 255) / np.float32(255) * 255) or abs(int(out[0, 0]) - 182) <= 1


def test_hsv_lab_matches_opencv_and_hue_range(bgr_image):
    out = cs.hsv_plus_lab(bgr_image)
    assert np.array_equal(out[..., :3], cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV))
    assert np.array_equal(out[..., 3:], cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB))
    assert out[..., 0].max() <= 179            # OpenCV 8-bit hue, so hue/255 tops out near 0.70


def test_log_a_channel_range(bgr_image):
    a = cs.rgb_plus_log_a(bgr_image)[..., 3]
    assert a.max() == 255 and a.min() >= 0


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        cs.get_colour_transform('ycbcr')


def _load_original(filename):
    path = os.path.join(ARCHIVE, filename)
    if not os.path.exists(path):
        pytest.skip('archive not present')
    spec = importlib.util.spec_from_file_location('orig_' + filename[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('filename,name,kwargs,channels', [
    ('dataset.py', 'rgb', {}, 3),
    ('dataset_RGB_L.py', 'rgb_y', {'power': 5}, 4),      # the archived file holds exponent 5
    ('dataset_RGB_A.py', 'rgb_a', {}, 4),
    ('dataset_RGB_HSV.py', 'hsv_lab', {}, 6),
])
def test_refactored_dataset_matches_original_code(tmp_path, filename, name, kwargs, channels):
    """Same file in, same tensor out, for the original Dataset class and the refactored one."""
    from dfuc_colour.dataset import Dataset
    rng = np.random.RandomState(1)
    img_dir, mask_dir = tmp_path / 'images', tmp_path / 'masks'
    img_dir.mkdir(); mask_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(img_dir / ('a%d.jpg' % i)), rng.randint(0, 256, (48, 64, 3)).astype(np.uint8))
        cv2.imwrite(str(mask_dir / ('a%d.png' % i)), (rng.rand(48, 64) > 0.5).astype(np.uint8) * 255)
    ids = ['a0', 'a1', 'a2']
    original = _load_original(filename).Dataset(ids, str(img_dir), str(mask_dir), '.jpg', '.png', 1)
    fn, _ = cs.get_colour_transform(name, **kwargs)
    refactored = Dataset(ids, str(img_dir), str(mask_dir), '.jpg', '.png', 1, colour_transform=fn)
    for i in range(3):
        a_img, a_mask, _ = original[i]
        b_img, b_mask, _ = refactored[i]
        assert b_img.shape == (channels, 48, 64)
        assert np.array_equal(a_img, b_img)
        assert np.array_equal(a_mask, b_mask)
