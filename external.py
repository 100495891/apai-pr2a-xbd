#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:07:52 2020

@author: mmolina
"""

from skimage.transform import resize
import numpy as np
import os
import torch
from PIL import Image

def get_features_hook0(self, input, output):
    features = output.data.cpu().numpy()
    np.save('f0/features.npy',features)
    
def get_features_hook1(self, input, output):
    features = output.data.cpu().numpy()
    np.save('f1/features.npy',features)
    
def get_features_hook2(self, input, output):
    features = output.data.cpu().numpy()
    np.save('f2/features.npy',features)

def get_features_hook3(self, input, output):
    features = output.data.cpu().numpy()
    np.save('f3/features.npy',features)
    
def get_activations(model, dataloaders, device, result_dir, batch_size=1):
    # Create the directories
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(os.path.join(result_dir,'activations')):
        os.mkdir(os.path.join(result_dir,'activations'))  
    # Create auxiliary directories for activations
    if not os.path.exists(os.path.join(os.getcwd(),'f0')):
        os.mkdir(os.path.join(os.getcwd(),'f0'))
    if not os.path.exists(os.path.join(os.getcwd(),'f1')):
        os.mkdir(os.path.join(os.getcwd(),'f1'))
    if not os.path.exists(os.path.join(os.getcwd(),'f2')):
        os.mkdir(os.path.join(os.getcwd(),'f2'))
    if not os.path.exists(os.path.join(os.getcwd(),'f3')):
        os.mkdir(os.path.join(os.getcwd(),'f3'))
    # Modify the model to obtain intermediate layers
    model.classifier.convs[0][3].register_forward_hook(get_features_hook0)
    model.classifier.convs[1][3].register_forward_hook(get_features_hook1)
    model.classifier.convs[2][3].register_forward_hook(get_features_hook2)
    model.classifier.convs[3][3].register_forward_hook(get_features_hook3)
    for sample in dataloaders['Test']:
        with torch.no_grad():
            inputs = sample['image'].to(device)
            prediction = model(inputs)
            os.rename(os.path.join(os.getcwd(),'f0','features.npy'),os.path.join(os.getcwd(),'f0',sample['img_path'][0].split(os.path.sep)[-1][:-4]+'.npy'))
            os.rename(os.path.join(os.getcwd(),'f1','features.npy'),os.path.join(os.getcwd(),'f1',sample['img_path'][0].split(os.path.sep)[-1][:-4]+'.npy'))
            os.rename(os.path.join(os.getcwd(),'f2','features.npy'),os.path.join(os.getcwd(),'f2',sample['img_path'][0].split(os.path.sep)[-1][:-4]+'.npy'))
            os.rename(os.path.join(os.getcwd(),'f3','features.npy'),os.path.join(os.getcwd(),'f3',sample['img_path'][0].split(os.path.sep)[-1][:-4]+'.npy'))
    
    for sample in dataloaders['Test']:
        with torch.no_grad():
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)
            prediction = model(inputs)
            y_pred = prediction['out']
            for j in range(y_pred.shape[0]):
                img=Image.open(sample['img_path'][j])
                img_name=sample['img_path'][j].split(os.path.sep)[-1]
                dataset_name=sample['img_path'][j].split(os.path.sep)[-2]
                # Analyze the features
                f0=np.load(os.path.join(os.getcwd(),'f0',sample['img_path'][0].split(os.path.sep)[-1][:-4]+'.npy'))
                f1=np.load(os.path.join(os.getcwd(),'f1',sample['img_path'][0].split(os.path.sep)[-1][:-4]+'.npy'))
                f2=np.load(os.path.join(os.getcwd(),'f2',sample['img_path'][0].split(os.path.sep)[-1][:-4]+'.npy'))
                f3=np.load(os.path.join(os.getcwd(),'f3',sample['img_path'][0].split(os.path.sep)[-1][:-4]+'.npy'))
                f0=resize(np.squeeze(f0[:,1,:,:]),img.size)
                f1=resize(np.squeeze(f1[:,1,:,:]),img.size)
                f2=resize(np.squeeze(f2[:,1,:,:]),img.size)
                f3=resize(np.squeeze(f3[:,1,:,:]),img.size)
                f0=f0-np.min(f0)
                f0=f0/np.max(f0)
                f0=Image.fromarray((255.0*f0).astype(np.uint8))
                f0.save(os.path.join(result_dir,'activations',img_name[:-4]+'_0.png'))
                f1=f1-np.min(f1)
                f1=f1/np.max(f1)
                f1=Image.fromarray((255.0*f1).astype(np.uint8))
                f1.save(os.path.join(result_dir,'activations',img_name[:-4]+'_1.png'))
                f2=f2-np.min(f2)
                f2=f2/np.max(f2)
                f2=Image.fromarray((255.0*f2).astype(np.uint8))
                f2.save(os.path.join(result_dir,'activations',img_name[:-4]+'_2.png'))
                f3=f3-np.min(f3)
                f3=f3/np.max(f3)
                f3=Image.fromarray((255.0*f3).astype(np.uint8))
                f3.save(os.path.join(result_dir,'activations',img_name[:-4]+'_3.png'))
