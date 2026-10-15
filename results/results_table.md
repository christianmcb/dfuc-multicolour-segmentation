## Validation (paper Table 2), best epoch by validation IoU, one run per row

| Colour space | Epoch | T-Loss | T-IoU | T-Dice | V-Loss | V-IoU | V-Dice | Saved-log check |
|---|---|---|---|---|---|---|---|---|
| RGB | 45 | 0.2371 | 0.7067 | 0.8193 | 0.4034 | 0.595 | 0.7341 | differs from baseline_bs8 (saved: epoch 67, val IoU 0.6019, val Dice 0.7394) |
| RGB + Y | 48 | 0.2272 | 0.7164 | 0.826 | 0.3987 | 0.5966 | 0.7354 | no saved log |
| RGB + A | 34 | 0.3024 | 0.6501 | 0.7782 | 0.4016 | 0.5827 | 0.7239 | no saved log |
| HSV + LAB | 44 | 0.2382 | 0.715 | 0.8247 | 0.4017 | 0.5957 | 0.7331 | matches hsv_lab_single_tensor |

## Test sets (paper Table 3), no saved output in this repository

**DFUC2022 test**

| Colour space | FNE | FPE | IoU | Dice | dIoU vs RGB | dDice vs RGB |
|---|---|---|---|---|---|---|
| RGB | 0.4313 | 0.3947 | 0.402 | 0.5181 | +0.0000 | +0.0000 |
| RGB + Y | 0.4038 | 0.3912 | 0.4176 | 0.5347 | +0.0156 | +0.0166 |
| RGB + A | 0.4481 | 0.3418 | 0.4121 | 0.5364 | +0.0101 | +0.0183 |
| HSV + LAB | 0.4107 | 0.3838 | 0.4207 | 0.5364 | +0.0187 | +0.0183 |

**CWDB**

| Colour space | FNE | FPE | IoU | Dice | dIoU vs RGB | dDice vs RGB |
|---|---|---|---|---|---|---|
| RGB | 0.2317 | 0.1871 | 0.6224 | 0.7376 | +0.0000 | +0.0000 |
| RGB + Y | 0.1959 | 0.1813 | 0.655 | 0.773 | +0.0326 | +0.0354 |
| RGB + A | 0.2313 | 0.1632 | 0.6362 | 0.7537 | +0.0138 | +0.0161 |
| HSV + LAB | 0.2168 | 0.1374 | 0.6698 | 0.7791 | +0.0474 | +0.0415 |

**AZH**

| Colour space | FNE | FPE | IoU | Dice | dIoU vs RGB | dDice vs RGB |
|---|---|---|---|---|---|---|
| RGB | 0.2787 | 0.4243 | 0.5208 | 0.6048 | +0.0000 | +0.0000 |
| RGB + Y | 0.2613 | 0.4264 | 0.512 | 0.5958 | -0.0088 | -0.0090 |
| RGB + A | 0.2551 | 0.4156 | 0.5231 | 0.6098 | +0.0023 | +0.0050 |
| HSV + LAB | 0.3069 | 0.4559 | 0.5003 | 0.5802 | -0.0205 | -0.0246 |

**FUSC**

| Colour space | FNE | FPE | IoU | Dice | dIoU vs RGB | dDice vs RGB |
|---|---|---|---|---|---|---|
| RGB | 0.4027 | 0.2571 | 0.4855 | 0.5948 | +0.0000 | +0.0000 |
| RGB + Y | 0.384 | 0.1929 | 0.5246 | 0.6379 | +0.0391 | +0.0431 |
| RGB + A | 0.4654 | 0.2201 | 0.463 | 0.5742 | -0.0225 | -0.0206 |
| HSV + LAB | 0.4008 | 0.2563 | 0.4982 | 0.6031 | +0.0127 | +0.0083 |

## DFUC2022 test by wound size (paper Table 4), no saved output in this repository

**DFUC2022 test, small wounds (<10%)**

| Colour space | FNE | FPE | IoU | Dice | dIoU vs RGB | dDice vs RGB |
|---|---|---|---|---|---|---|
| RGB | 0.7878 | 0.4213 | 0.1956 | 0.2628 | +0.0000 | +0.0000 |
| RGB + Y | 0.7752 | 0.4264 | 0.1954 | 0.2659 | -0.0002 | +0.0031 |
| RGB + A | 0.8066 | 0.378 | 0.173 | 0.2447 | -0.0226 | -0.0181 |
| HSV + LAB | 0.7764 | 0.4205 | 0.1997 | 0.2715 | +0.0041 | +0.0087 |

**DFUC2022 test, medium wounds (10-90%)**

| Colour space | FNE | FPE | IoU | Dice | dIoU vs RGB | dDice vs RGB |
|---|---|---|---|---|---|---|
| RGB | 0.4079 | 0.3212 | 0.4464 | 0.5633 | +0.0000 | +0.0000 |
| RGB + Y | 0.3762 | 0.317 | 0.4672 | 0.5845 | +0.0208 | +0.0212 |
| RGB + A | 0.426 | 0.2592 | 0.4566 | 0.5787 | +0.0102 | +0.0154 |
| HSV + LAB | 0.3821 | 0.3119 | 0.4681 | 0.5833 | +0.0217 | +0.0200 |

**DFUC2022 test, large wounds (>90%)**

| Colour space | FNE | FPE | IoU | Dice | dIoU vs RGB | dDice vs RGB |
|---|---|---|---|---|---|---|
| RGB | 0.1108 | 0.3953 | 0.5569 | 0.6744 | +0.0000 | +0.0000 |
| RGB + Y | 0.097 | 0.4017 | 0.5548 | 0.6694 | -0.0021 | -0.0050 |
| RGB + A | 0.1133 | 0.3324 | 0.6058 | 0.7196 | +0.0489 | +0.0452 |
| HSV + LAB | 0.1011 | 0.3593 | 0.5891 | 0.7035 | +0.0322 | +0.0291 |

dIoU / dDice are differences between paper values, computed by this script.

## Saved logs of selected runs (best epoch by validation IoU)

| run | epochs_run | best_epoch | val_loss | val_iou | val_dice_score |
|---|---|---|---|---|---|
| baseline_bs2 | 1000 | 45 | 0.3976 | 0.5603 | 0.6884 |
| baseline_bs4 | 59 | 38 | 0.3973 | 0.5694 | 0.7074 |
| baseline_bs8 | 88 | 67 | 0.4108 | 0.6019 | 0.7394 |
| baseline_bs16 | 62 | 41 | 0.3954 | 0.5857 | 0.7285 |
| hsv_lab_single_tensor | 65 | 44 | 0.4017 | 0.5957 | 0.7331 |
| rgb_a_single_tensor | 76 | 55 | 0.4008 | 0.5951 | 0.7335 |
| rgb_log_a | 55 | 34 | 0.4131 | 0.5772 | 0.7179 |
| rgb_luminance_1 | 69 | 48 | 0.4031 | 0.6075 | 0.7441 |
| rgb_luminance_2 | 76 | 55 | 0.3936 | 0.5988 | 0.7364 |
| rgb_luminance_3 | 76 | 55 | 0.3832 | 0.6101 | 0.746 |
| rgb_luminance_4 | 85 | 64 | 0.3878 | 0.6117 | 0.7466 |
| rgb_luminance_5 | 69 | 48 | 0.3982 | 0.594 | 0.7333 |
