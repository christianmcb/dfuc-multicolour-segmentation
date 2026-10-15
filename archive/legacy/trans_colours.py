"""
Python script to transform the colourspace of DFU segmentation images 
and save in new directory
"""

# Import specific libraries for conversion
import cv2
import os
import argparse
import shutil

# Define arguments (directories, colourspaces and which folders are being transformed)
def parse_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--train_dir', default='./datasets/dfuc2022/train/images', help='Train images directory: (default: ./datasets/dfuc2022/train/images)')
	
	parser.add_argument('--val_dir', default='./datasets/dfuc2022/val/images', help='Val images directory: (default: ./datasets/dfuc2022/val/images)')
	
	parser.add_argument('--test_dir', default='./datasets/dfuc2022/test', help='Test images directory: (default: ./datasets/dfuc2022/test)')
	
	parser.add_argument('--colourspace', default=[cv2.COLOR_BGR2LAB], help='List of OpenCV colourspaces to transform.') 
	
	parser.add_argument('--train_val_test', default=['train','val','test'], help='List of train/val/test to transform. (strings)') 
	
	config = parser.parse_args()
	return config
        

# Get the directories of images within files.
def walk_dirs(directory):
	for root, dirs, files in os.walk(directory):
		files = sorted([f'{directory}/{file}' for file in files], key=str)
	print('Saved {} files in directory {}'.format(len(files), directory))
	return files

def change_colourspace(directories, colourspace, train_val_test):
	for space in colourspace:
		print('Working on colourspace number {}....'.format(str(space)))
		
		path = 'dfuc2022_' + str(space) + '_' + '_'.join(i for i in train_val_test)
		
		# Create directories if they do not already exist.
		if not os.path.exists(path):
			os.makedirs(os.path.join('datasets', path))
			print('Directory {} created.'.format(path))
		
		for tvt in train_val_test:
			if tvt in ['train','val']:
				tvt_path = os.path.join('datasets', path, tvt, 'images')
			else:
				tvt_path = os.path.join('datasets', path, tvt)
				
			if not os.path.exists(tvt_path):
				os.makedirs(tvt_path)
				print('Directory {} created.'.format(tvt_path))
			
		for dirs in directories:
			for d in dirs:
				# Colourspace conversion
				img = cv2.imread(d)
				img = cv2.cvtColor(img, space)
				di = d.replace('dfuc2022/', path + '/')
				cv2.imwrite(di, img)
		print('Images saved to {} folder.'.format(path))


def main():
	# Retrieve arguments and print configurations
	config = vars(parse_args())
	for key in config:
		print('%s: %s' % (key, config[key]))

	train_test_val, directories, masks = [], [], []
	
	if config['train_dir'] is not None:
		train_files = walk_dirs(config['train_dir'])
		directories.append(train_files)
		train_test_val.append('train')
		
	if config['val_dir'] is not None:
		val_files = walk_dirs(config['val_dir'])
		directories.append(val_files)
		train_test_val.append('val')
		
	if config['test_dir'] is not None:
		test_files = walk_dirs(config['test_dir'])
		directories.append(test_files)
		train_test_val.append('test')
	
	# Run function to change colourspace of images and save to new directory
	change_colourspace(directories, config['colourspace'], train_test_val)


if __name__ == '__main__':
    main()
	
