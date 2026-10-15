import os

import cv2
import numpy as np
import torch
import torch.utils.data


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
        
        # Add HSV channels to original RGB image array
        imageHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        i1,i2,i3 = cv2.split(img)
        j1,j2,j3 = cv2.split(imageHSV)
        
        # Add LAB channels to original RGB image array
        imageLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        k1,k2,k3 = cv2.split(imageLAB)
        
        l = (i1*0.0722 + i2*0.7152 + i3*0.2126)
        l = np.array(((l) / np.max(l)) * 255, dtype='float32')
        
        luminance = np.array((l**5 / np.max(l**5)) * 255, dtype='uint8')
        
        img = cv2.merge((i1,i2,i3,luminance))
        

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
        
        #img = np.divide(img, np.array([179,255,255])).astype('float32')
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}
