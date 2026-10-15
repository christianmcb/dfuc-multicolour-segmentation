import glob
import ntpath

# script to verify that for every image there is a corresponding mask (and vice versa)

print('searching for missing masks...')

image_files = glob.glob("./test/images/*.jpg")
mask_files = glob.glob("./test/masks_snake/*.png")

image_filenames = []
for image in image_files:
  path, filename = ntpath.split(image)
  filename, extension = filename.split('.')
  image_filenames.append(filename)

mask_filenames = []
for mask in mask_files:
  path, filename = ntpath.split(mask)
  filename, extension = filename.split('.')
  mask_filenames.append(filename)

missing_masks = []
for file in image_filenames:
  if file not in mask_filenames:
    missing_masks.append(file)

print('searching for missing images...')

missing_images = []
for file in mask_filenames:
  if file not in image_filenames:
    missing_images.append(file)

if (len(missing_masks) > 0):
  print('missing masks:')
  for file in missing_masks:
    print(file)
else:
  print('no missing masks')

if (len(missing_images) > 0):
  print('missing images:')
  for file in missing_images:
    print(file)
else:
  print('no missing images')
