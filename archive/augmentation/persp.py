import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import ntpath

# perspective warp with zoom (image and mask)

# uses code from here: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

from glob import glob
from PIL import Image, ImageOps

src_images = glob('<DATA_ROOT>/inputs/dfuc2022/train/images/*')
images_output_folder_path = '<DATA_ROOT>/inputs/dfuc2022_persp/train/images/'
masks_output_folder_path = '<DATA_ROOT>/inputs/dfuc2022_persp/train/masks/'

if not os.path.isdir(images_output_folder_path):
  os.mkdir(images_output_folder_path)
  
if not os.path.isdir(masks_output_folder_path):
  os.mkdir(masks_output_folder_path)

# apply same transform to image and corresponding mask
# note: you need to account for images with completely black masks that result from this processing
# best option is to delete them (mask and image pairs) after running this script
for img_path in src_images:
  path, filename = ntpath.split(img_path)
  filename, extension = filename.split('.')
  print('processing: ' + filename)
  img = cv2.imread(img_path)
  msk = cv2.imread('<DATA_ROOT>/inputs/dfuc2022_persp/train/masks/' + filename + '.png')
  rows,cols,ch = img.shape
  pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
  pts2 = np.float32([[20,0],[600,0],[10,600],[600,600]])
  M = cv2.getPerspectiveTransform(pts1,pts2)
  image_output = cv2.warpPerspective(img,M,(-100,-100))
  mask_output = cv2.warpPerspective(msk,M,(-100,-100))
  cv2.imwrite(os.path.join(images_output_folder_path, 'persp_' + filename + '.jpg'), image_output)

  # convert mask to black and white pixels
  grayImage = cv2.cvtColor(mask_output, cv2.COLOR_BGR2GRAY)
  (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
  cv2.imwrite(os.path.join(masks_output_folder_path, 'persp_' + filename + '.png'), blackAndWhiteImage)

# _, (ax1) = plt.subplots(1)
# ax1.imshow(img)
# plt.show()
