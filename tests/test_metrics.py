import pytest
import torch

from dfuc_colour.losses import BCEDiceLoss
from dfuc_colour.metrics import dice_coef, iou_score


def logits(mask, big=20.0):
    """Saturated logits so sigmoid is ~0 or ~1: a binary prediction."""
    return (mask.float() * 2 - 1) * big


def make(rows):
    return torch.tensor(rows, dtype=torch.float32).reshape(1, 1, *torch.tensor(rows).shape)


TARGET = make([[1, 1, 0, 0], [1, 1, 0, 0]])           # 4 foreground pixels


def test_perfect_prediction():
    assert iou_score(logits(TARGET), TARGET) == pytest.approx(1.0, abs=1e-4)
    assert dice_coef(logits(TARGET), TARGET) == pytest.approx(1.0, abs=1e-4)


def test_hand_computed_partial_overlap():
    # prediction: 3 of the 4 target pixels plus 1 wrong pixel. intersection 3, union 5, |P|=|T|=4
    pred = make([[1, 1, 0, 1], [1, 0, 0, 0]])
    assert iou_score(logits(pred), TARGET) == pytest.approx(3 / 5, abs=1e-4)
    assert dice_coef(logits(pred), TARGET) == pytest.approx(2 * 3 / (4 + 4), abs=1e-4)


def test_no_overlap():
    pred = make([[0, 0, 1, 1], [0, 0, 1, 1]])
    assert iou_score(logits(pred), TARGET) == pytest.approx(0.0, abs=1e-4)
    assert dice_coef(logits(pred), TARGET) == pytest.approx(0.0, abs=1e-4)


def test_both_empty_scores_one_because_of_smoothing():
    empty = torch.zeros(1, 1, 2, 4)
    assert iou_score(logits(empty), empty) == pytest.approx(1.0, abs=1e-4)
    # soft Dice: sigmoid(-20) is ~2e-9 per pixel, not exactly 0, so it lands just below 1
    assert dice_coef(logits(empty), empty) == pytest.approx(1.0, abs=5e-3)


def test_iou_threshold_is_half_on_the_probability():
    # logit 0 -> p = 0.5, not > 0.5, so the pixel is background; dice (soft) still counts 0.5
    out = torch.zeros(1, 1, 1, 2)
    tgt = torch.tensor([[[[1.0, 0.0]]]])
    assert iou_score(out, tgt) == pytest.approx(0.0, abs=1e-4)
    assert dice_coef(out, tgt) == pytest.approx(2 * 0.5 / (1.0 + 1.0), abs=1e-4)


def test_batch_iou_is_pooled_over_the_batch_not_a_mean_of_images():
    # image 1: perfect on a 1-pixel wound; image 2: wrong on a 3-pixel wound. Pooled IoU = intersection / union
    t = torch.tensor([[[[1., 0., 0., 0.]]], [[[1., 1., 1., 0.]]]])
    p = torch.tensor([[[[1., 0., 0., 0.]]], [[[0., 0., 0., 1.]]]])
    pooled = iou_score(logits(p), t)
    assert pooled == pytest.approx(1 / 5, abs=1e-4)           # intersection 1; union = 1 (image 1) + 3 + 1 stray (image 2)
    mean_of_images = (1.0 + 0.0) / 2
    assert pooled != pytest.approx(mean_of_images, abs=1e-2)


def test_bce_dice_loss_hand_value():
    # logits 0 everywhere, empty target: BCE = ln 2, soft dice ~ 0 so (1 - dice) ~ 1; loss = 0.5 ln 2 + 1
    x = torch.zeros(1, 1, 4, 4)
    y = torch.zeros(1, 1, 4, 4)
    expected = 0.5 * torch.log(torch.tensor(2.0)) + 1.0
    assert BCEDiceLoss()(x, y).item() == pytest.approx(expected.item(), abs=1e-4)
