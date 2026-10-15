import pytest
import torch

from dfuc_colour.models import UNet


@pytest.mark.parametrize('channels', [3, 4, 6])
def test_unet_output_shape(channels):
    model = UNet(num_classes=1, input_channels=channels).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, channels, 48, 64))
    assert out.shape == (2, 1, 48, 64)           # logits, one per pixel; no sigmoid inside the model


def test_only_first_layer_depends_on_input_channels():
    a = UNet(1, 3).state_dict()
    b = UNet(1, 6).state_dict()
    differing = [k for k in a if a[k].shape != b[k].shape]
    assert differing == ['conv0_0.conv1.weight']
