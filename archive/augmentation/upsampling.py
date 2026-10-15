import os
import skimage
import imageio as io
import matplotlib.pyplot as plt
import random
import cv2

from glob import glob
from skimage.io import imsave, imread

# apply upsampling to images

src_images = glob('<DATA_ROOT>/inputs/dfuc2022/val/images/*')
output_folder_path = '<DATA_ROOT>/inputs/dfuc2022_upsampling/val/images/'

if not os.path.isdir(output_folder_path):
  os.mkdir(output_folder_path)
  
for img_path in src_images:
  drive, path_and_file = os.path.splitdrive(img_path)
  path, file = os.path.split(path_and_file)
  img = cv2.pyrUp(imread(img_path))
  
  print(output_folder_path + file)
  imsave(output_folder_path + 'upsample_' + file, img)
