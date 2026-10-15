import matplotlib.pyplot as plt
import cv2
import os
import random
import ntpath

from glob import glob
from skimage.io import imsave, imread

# random flip
flips = [-1, 0, 1] # -1: both, 0: vertical, 1: horizontal


src_images = glob('<DATA_ROOT>/inputs/dfuc2022/train/images/*')
src_masks = glob('<DATA_ROOT>/inputs/dfuc2022/train/masks/*')
output_image_path = '<DATA_ROOT>/inputs/dfuc2022_random_flip/train/images/'
output_mask_path = '<DATA_ROOT>/inputs/dfuc2022_random_flip/train/masks/'

if not os.path.isdir(output_image_path):
  os.mkdir(output_image_path)
  
if not os.path.isdir(output_mask_path):
  os.mkdir(output_mask_path)
  
for img_path in src_images:
  img_drive, img_path_and_file = os.path.splitdrive(img_path)
  img_path2, img_file = os.path.split(img_path_and_file)
  
  mask_path = img_path.replace('images','masks').replace('jpg','png')
  
  print(img_path, mask_path)
  
  # random flip - apply same flip to image and corresponding mask
  flip_type = random.choice(flips)
  img = cv2.flip(cv2.imread(img_path), flip_type)
  mask = cv2.flip(cv2.imread(mask_path), flip_type)
  
  print(output_image_path + img_file)
  cv2.imwrite(output_image_path + 'flip_' + img_file , img)
  cv2.imwrite(output_mask_path + 'flip_' + img_file.replace('jpg','png'), mask)
