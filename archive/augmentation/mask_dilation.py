import os
import skimage
import imageio as io
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np

from glob import glob
from skimage.io import imsave, imread

# apply upsampling to images

src_images = glob('<DATA_ROOT>/inputs/dfuc2022/train/images/*')
output_folder_path = '<DATA_ROOT>/inputs/dfuc2022_sharpen/train/images/'

if not os.path.isdir(output_folder_path):
  os.mkdir(output_folder_path)
  
for img_path in src_images:
  drive, path_and_file = os.path.splitdrive(img_path)
  path, file = os.path.split(path_and_file)
  img = cv2.filter2D(imread(img_path), -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
  
  print(output_folder_path + file)
  imsave(output_folder_path + 'sharpen_' + file, img)
