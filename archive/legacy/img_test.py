import cv2
import os
import matplotlib
import matplotlib.pyplot as plt

# Get the directories of images within files.
def walk_dirs(directory):
	for root, dirs, files in os.walk(directory):
		files = sorted([f'{directory}/{file}' for file in files], key=str)
	print('Saved {} files in directory {}'.format(len(files), directory))
	return files
	
train_files = walk_dirs('./datasets/dfuc2022/train/images')

img = cv2.imread(train_files[0])
plt.imshow(img)
plt.show()
