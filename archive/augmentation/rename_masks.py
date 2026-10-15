import glob
import os
import ntpath

# append a prefix to mask files - used when augmenting an image set where masks don't 
# require transformation and only need to copy original masks

images = glob.glob('<DATA_ROOT>/inputs/dfuc2022_cutout/train/masks/*')

for file in images:
  path, filename = ntpath.split(file)
  filename, extension = filename.split('.')
  fname = 'rand_erase_' + filename + '.png'
  os.rename(file, path + '/' + fname)
