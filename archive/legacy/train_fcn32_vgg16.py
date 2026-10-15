
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:38:01 2021

@author: 55114405
"""

import tensorflow as tf
from tensorflow import keras 
import keras_segmentation
import keras_segmentation.models
import keras_segmentation.models.fcn
import keras_segmentation.models.unet
import keras_segmentation.models.pspnet
import keras_segmentation.models.segnet
import numpy as np
import os


tf.random.set_seed(42)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())
print(tf.test.gpu_device_name())
print(keras.__version__)
print(tf.__version__)
print(np.__version__)

# https://keras.io/api/applications/
#'fcn_8_mobilenet','fcn_32_mobilenet','pspnet','vgg_pspnet','resnet50_pspnet',
# array of models we are  training
model_to_train = ['fcn_32'] #['fcn_32_vgg']
# models_to_train = ['fcn_32','fcn_8_vgg','fcn_32_vgg',
#                   'fcn_8_resnet50','fcn_32_resnet50',
#                   'unet_mini','unet','vgg_unet','resnet50_unet',
#                   'segnet','vgg_segnet','resnet50_segnet']# 'unet'


# this is wrong
optimisers = ['adadelta']
batch_sizes = [2]


class_count = 2
training_loops = 1
# max_size = 512
save_path = r'<DATA_ROOT>/save'
num_gpus = 1
num_epochs = 2000
# set this to imagenet to pretrain on that
weights = None #"imagenet"
train_img_count = 1600
val_img_count = 400

# In[]: standard model versions

for model_count in range(len(models_to_train)):
    model_type = models_to_train[model_count] 
    if not os.path.exists(os.path.join(save_path,model_type)):
            os.mkdir(os.path.join(save_path,model_type))
    currSavePath = os.path.join(save_path,model_type)
    for opt_count in range(len(optimisers)):
        opt_type = optimisers[opt_count]
        if not os.path.exists(os.path.join(currSavePath,opt_type)):
            os.mkdir(os.path.join(currSavePath,opt_type))
        currSavePath = os.path.join(currSavePath,opt_type)
        model = None 
        for batchSizes in range (len(batch_sizes)): 
            batch = batch_sizes[batchSizes]
            tf.keras.backend.clear_session()
            if not os.path.exists(os.path.join(currSavePath,str(batch))):
                os.mkdir(os.path.join(currSavePath,str(batch)))
            currSavePath = os.path.join(currSavePath,str(batch))
            
            
            csv_logger =  [
                    tf.keras.callbacks.CSVLogger(
                        os.path.join(currSavePath,model_type+'_batchSize_'+str(batch)+"_opt_"+opt_type+".csv"), separator=",", append=False),
                    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(currSavePath,model_type+'_batchSize_'+str(batch)+"_opt_"+opt_type+'_'+'model.h5'),
                                                       save_best_only=True,
                                                       monitor="val_accuracy",
                                                       mode='max',
                                                       save_weights_only=False),
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_accuracy",
                        min_delta=0,
                        patience=10,
                        verbose=0,
                        mode="max",
                        baseline=None,
                        restore_best_weights=False,)]
            # build model
            opt_func = None 
            if("unet" in model_type):
                opt_func = getattr(keras_segmentation.models.unet,model_type)
            elif("fcn" in model_type): 
                opt_func = getattr(keras_segmentation.models.fcn,model_type)
            elif("segnet" in model_type): 
                opt_func = getattr(keras_segmentation.models.segnet,model_type)
            elif("pspnet" in model_type): 
                opt_func = getattr(keras_segmentation.models.pspnet,model_type)
            model =  opt_func(n_classes=2 ,  input_height=480, input_width=640)
            model.train(
                train_images =  "<DATA_ROOT>/dfuc2022/train/images",
                train_annotations = "<DATA_ROOT>/dfuc2022/train/masks",
                validate=True, 
                val_images="<DATA_ROOT>/dfuc2022/val/images",
                val_annotations="<DATA_ROOT>/dfuc2022/val/masks",
                checkpoints_path = "<DATA_ROOT>/checkpoints/checkpoints", 
                epochs=num_epochs,
                batch_size=batch,
                val_batch_size=batch, 
                callbacks=csv_logger,
                optimizer_name = optimisers[opt_count],
                steps_per_epoch=train_img_count//batch,
                val_steps_per_epoch=val_img_count//batch,
                do_augment = True,
            )

# In[]: 
    
#import cv2 as cv
#import os 

#path = "C:/Users/conna/Documents/img_datasets/DFU2022/val/val/validation_mask/"

#images = os.listdir(path)

#for image in images: 
#    img = cv.imread(path+image,0)
#    img[img > 0] = 1
#    cv.imwrite(path+image,img)
