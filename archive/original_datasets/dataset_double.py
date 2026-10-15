import os

import cv2
import numpy as np
import torch
import torch.utils.data
from scripts.shades_of_gray import shade_of_gray_cc
from scripts.random_eraser import get_random_eraser


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            # mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
            #             img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
            mask.append(cv2.imread(os.path.join(self.mask_dir,
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # Sharpen augmentation
        # image = cv2.filter2D(img, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
        
        # Shades of gray augmentation
        #image = shade_of_gray_cc(img)
        
        # Random eraser augmentation
        eraser = get_random_eraser(
				  p=1.0,
				  s_l=0.07,
				  s_h=0.16,
				  r_1=0.3,
				  r_2=1/0.3,
				  v_l=0,
				  v_h=255,
				  pixel_level=True)
        image = eraser(img)
        
        
        i1,i2,i3 = cv2.split(img)
        j1,j2,j3 = cv2.split(image)
        
        img = cv2.merge((i1,i2,i3,j1,j2,j3))
        
        #img = np.divide(img, np.array([179,255,255])).astype('float32')
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}
