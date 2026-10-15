import ntpath
import glob
import cv2

# script to convert images to JPG format at 50% compression
# files are converted in-place, so make sure you backup your images first!

images = glob.glob("<DATA_ROOT>/inputs/dfuc2022/train/images/*")
output_dir = '<DATA_ROOT>/inputs/dfuc2022_jpg50/train/images/'

print('compressing images...')

for file in images:
  path, filename = ntpath.split(file)
  filename, extension = filename.split('.')
  img = cv2.imread(file)
  cv2.imwrite(output_dir + 'jpg50_' + filename + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
  
print('process complete')
