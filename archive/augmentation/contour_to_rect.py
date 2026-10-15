import glob
import cv2
import ntpath
import numpy as np

# this script will take a full size image and its mask and crop the region of interest into 
# a rectangle based on the mask, resize the cropped image and mask to 128x128 then save the
# new image and mask
#
# it can handle multiple mask regions per single image

print('processing contours...')
images = glob.glob("./val/masks_snake/*.png")

for file in images:
  path, filename = ntpath.split(file)
  filename, extension = filename.split('.')
  o = cv2.imread('val/images/' + filename + '.jpg') # image
  v = cv2.imread(file) # mask
  s = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
  s = cv2.Laplacian(s, cv2.CV_16S, ksize=3)
  s = cv2.convertScaleAbs(s)

  ret, binary = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
  i = 1
  for c in contours:
    # cv2.findContours returns two slight slightly different sets of contours for each single mask
	# so we need to just get one here
    if (i % 2) != 0: # i is not even
      i += 1
      continue
    x, y, w, h = cv2.boundingRect(c)
    # cv2.rectangle(v, (x,y), (x+w,y+h), (0,0,0), 0)
	# crop, resize, and save wound image
    crop_img = o[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img, (128,128), interpolation=cv2.INTER_AREA)
    cv2.imwrite('val_images/' + filename + '_' + str(i) + '.jpg', crop_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
	# crop, resize, and save mask
    crop_img = v[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img, (128,128), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('val_masks_snake/' + filename + '_' + str(i) + '.png', crop_img)
    i += 1
  
  # cv2.imshow('output', v)
  # cv2.waitKey()
  # cv2.destroyAllWindows()

print('finished')
