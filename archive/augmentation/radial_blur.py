import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import imutils

from glob import glob
from PIL import Image, ImageOps

# uses code from here: https://stackoverflow.com/questions/7607464/implement-radial-blur-with-opencv
 
src_images = glob('<DATA_ROOT>/inputs/dfuc2022/train/images/*')
output_folder_path = '<DATA_ROOT>/inputs/dfuc2022_radialBlur/train/images/'
desired_size = (640, 480)

if not os.path.isdir(output_folder_path):
  os.mkdir(output_folder_path)
  
for img_path in src_images:
  drive, path_and_file = os.path.splitdrive(img_path)
  path, file = os.path.split(path_and_file)
  print('processing: ' + file)
  
  img = cv2.imread(img_path)
  
  # apply radial blur
  w, h = img.shape[:2]
  center_x = w / 2
  center_y = h / 2
  blur = 0.01
  iterations = 6
  growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
  shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
  growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
  shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)

  for i in range(iterations):
    tmp1 = cv2.remap(img, growMapx, growMapy, cv2.INTER_LINEAR)
    tmp2 = cv2.remap(img, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
    img = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)

  # center crop to remove border created by blur
  border = 6 # might need to increase this for images larger than 224x224
  img = img[border:-border, border:-border]
  
  # resize back to 224x224 to compensate for crop (INTER_LINEAR preferred for enlarging)
  img = cv2.resize(img, desired_size, interpolation = cv2.INTER_LINEAR)
  
  # save image
  cv2.imwrite(os.path.join(output_folder_path, 'radblur_' + file), img)

# _, (ax1) = plt.subplots(1)
# ax1.imshow(img)
# plt.show()
