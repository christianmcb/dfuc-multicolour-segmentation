import os
import skimage
import imageio as io
import matplotlib.pyplot as plt
import random

from glob import glob
from skimage.io import imsave, imread

# apply random noise to images
# uses code from: https://gist.github.com/Prasad9/28f6a2df8e8d463c6ddd040f4f6a028a

src_images = glob('<DATA_ROOT>/inputs/dfuc2022/train/images/*')
output_folder_path = '<DATA_ROOT>/inputs/dfuc2022_noise/train/images/'

if not os.path.isdir(output_folder_path):
  os.mkdir(output_folder_path)
  
for img_path in src_images:
  drive, path_and_file = os.path.splitdrive(img_path)
  path, file = os.path.split(path_and_file)
  img = imread(img_path)/255.0
  
  noise_modes = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']  
  noise_mode = noise_modes[random.randint(0, 6)]
  img = skimage.util.random_noise(img, mode=noise_mode)
  print(output_folder_path + file)
  imsave(output_folder_path + 'noise_' + file, img)
  # img = cv2.imread(img_path)
  # vflipped = cv2.flip(img, 0)
  # cv2.imwrite(os.path.join(output_folder_path, 'vflip_' + file), vflipped)

def plotnoise(img, mode, r, c, i):
  plt.subplot(r,c,i)
  if mode is not None:
    gimg = skimage.util.random_noise(img, mode=mode)
    plt.imshow(gimg)
  else:
    plt.imshow(img)
  plt.title(mode)
  plt.axis("off")
