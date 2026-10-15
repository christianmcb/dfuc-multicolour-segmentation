import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from glob import glob
from random_eraser import get_random_eraser

# uses code from here: https://github.com/yu4u/cutout-random-erasing

# p : the probability that random erasing is performed
# s_l, s_h : minimum / maximum proportion of erased area against input image
# r_1, r_2 : minimum / maximum aspect ratio of erased area
# v_l, v_h : minimum / maximum value for erased area
# pixel_level : pixel-level randomization for erased area
eraser = get_random_eraser(
  p=1.0,
  s_l=0.07,
  s_h=0.16,
  r_1=0.3,
  r_2=1/0.3,
  v_l=0,
  v_h=255,
  pixel_level=True)
  
src_images = glob('<DATA_ROOT>/inputs/dfuc2022/train/images/*') # then do again with *.JPG
output_folder_path = '<DATA_ROOT>/inputs/dfuc2022_cutout/train/images/'

if not os.path.isdir(output_folder_path):
  os.mkdir(output_folder_path)
  
for img_path in src_images:
  drive, path_and_file = os.path.splitdrive(img_path)
  path, file = os.path.split(path_and_file)
  print('processing: ' + file)
  img = cv2.imread(img_path)
  img = eraser(img)
  cv2.imwrite(os.path.join(output_folder_path, 'rand_erase_' + file), img)

# _, (ax1) = plt.subplots(1)
# ax1.imshow(img)
# plt.show()
