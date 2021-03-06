#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:33:38 2022

@author: bougourzi
"""



import torch.nn.functional as F
# Data_loader2 for Lung &  Data_loader3 for Infection
from Data_loader import Data_loader2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as TF
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

############################################
############################################

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import nibabel as nib
from sklearn.model_selection import train_test_split

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)



img_size = 224
batch_size = 6
epochs = 60


import os
import cv2
import torch

import numpy as np
import re
from sklearn.model_selection import train_test_split
############################
# Part 1
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

############################
def reverse_transformrgb(inp):
    inp = inp.squeeze(dim=0).cpu().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


def reverse_transform(inp):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp
############################
segm = nn.Sigmoid()
############################

#####################################
test_transformfilt = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),          
]) 

val_transforms = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


model_lung = './ModelsAtt2/Model_AttUnet2_LungSeg2_data2_60epochs_bce224_bt2.pt'
model_inf =  './ModelsAtt2/Model_AttUNet2_infSeg2_data2_60epochs_bce_bt.pt'
model_filt =  './Models/Rex_best.pt'
device = torch.device("cuda:0")

print('done first')

import PYNetworks as networks
# model1 = networks.AttUNet().to(device)
# model1.load_state_dict(torch.load(model_inf))

model2 = networks.AttUNet().to(device)
model2.load_state_dict(torch.load(model_lung))

model3 =  torchvision.models.resnext50_32x4d(pretrained=True) 
model3.fc =  nn.Linear(2048, 2)
model3.load_state_dict(torch.load(model_filt))
model3.to(device)

print('done load models')
############
database_path = '/data/datach/Train/'
images_path = 'Covid'
    
Train_save_path_Slice = "./NumpysSeg4/Train/"
if not os.path.exists(Train_save_path_Slice):
    os.makedirs(Train_save_path_Slice) 
    
kernel = np.ones((5,5),np.uint8)
    
labels = []

# Part 4
data_splits = sorted_alphanumeric(os.listdir(os.path.join(database_path, images_path)))

kk = data_splits
tr_idx = -1

imgs_lst = []
for split in kk:
    split_dir = os.path.join(database_path, images_path, split)
    images_names = sorted_alphanumeric(os.listdir(split_dir)) 
         
    imgs_lst = []
    lung_lst = []
    lung_lst2 =[]
    print(len(images_names))
    for image in images_names:
        # tr_idx += 1
        try:
            
            im_path = os.path.join(database_path, images_path, split, image)            
            img = cv2.imread(im_path)
            # print("pass")
            imfil = test_transformfilt(img)
            imfil = imfil.float().unsqueeze(dim=0).to(device)
            # print(imfil.shape)
            model3.eval()
            with torch.no_grad():
                pred = model3(imfil)
            
            pred = pred.argmax(dim=1)
                
            # print(pred)
                
            if pred == 1:                
                augmentations = val_transforms(image=img)
                img1 = augmentations["image"]
                       
                test_img = img1.float().to(device).unsqueeze(dim=0)
                        
                # model1.eval()
                model2.eval()
                with torch.no_grad():
                    # pred1 = segm(model1(test_img))
                    pred2 = segm(model2(test_img)) 
                    
                predb2 = pred2 > 0.5
                predb2 = predb2.squeeze(dim=1)
                mask_pred = reverse_transform(predb2)
                # print(mask_pred.shape)
                mask_pred[mask_pred > 0.0] = 1.0
                y33 = mask_pred*255.0
                # print(y33.shape)
                y33 = y33.astype(np.uint8)
                y3 = cv2.dilate(y33,kernel,iterations = 1)
                # y3 = cv2.morphologyEx(y33, cv2.MORPH_GRADIENT, kernel)
                y3 = np.expand_dims(y3, 2)
                # y3 =  mask_pred+y3 
                # print(y3.shape)
                y3[y3 > 0.0] = 1.0
                # y3 = np.expand_dims(y3, 2)
                y3 = cv2.resize(y3, (512,512), interpolation = cv2.INTER_AREA)
                y3 = np.stack((y3, y3, y3), axis=2)
                
                lung_img = y3*img
                lung_img = cv2.cvtColor(lung_img, cv2.COLOR_BGR2GRAY)
                # lung_img = y3*cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
                # lung_img = clahe.apply(lung_img)
                equ = cv2.equalizeHist(lung_img)                   
            
                gray = cv2.cvtColor(cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                # gray2 = cv2.cvtColor(cv2.resize(lung_img, (224,224), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                imgs_lst.append(np.expand_dims(gray, 2)) 
                lung_lst.append(np.expand_dims(lung_img, 2)) 
                lung_lst2.append(np.expand_dims(equ, 2))

        except:
            pass
        
    if len(imgs_lst) >3:
        tr_idx += 1
        # print(tr_idx)
        labels.append(1)
        imgs_lst1 = np.concatenate(imgs_lst, axis=2)
        np.save(os.path.join(Train_save_path_Slice, str(tr_idx)+'.npy'), imgs_lst1) 
        
        lung_lst1 = np.concatenate(lung_lst, axis=2)
        np.save(os.path.join(Train_save_path_Slice, str(tr_idx)+'_lung.npy'), lung_lst1) 
        
        lung_lst2 = np.concatenate(lung_lst2, axis=2)
        np.save(os.path.join(Train_save_path_Slice, str(tr_idx)+'_lung2.npy'), lung_lst2)         
               
        
####################################################################################

    

############
# Part 4
############
database_path = '/data/datach/Train/'
images_path = 'Non-Covid'
    
Train_save_path_Slice = "./NumpysSeg4/Train/"
if not os.path.exists(Train_save_path_Slice):
    os.makedirs(Train_save_path_Slice) 
    
    
# labels = [][0:10]

# Part 4
data_splits = sorted_alphanumeric(os.listdir(os.path.join(database_path, images_path)))
kk = data_splits
# tr_idx = -1
for split in kk:
    split_dir = os.path.join(database_path, images_path, split)
    images_names = sorted_alphanumeric(os.listdir(split_dir)) 
    print(len(images_names))
         
    imgs_lst = []
    inf_lst = []
    lung_lst = []
    lung_lst2 = []
    for image in images_names:
        # tr_idx += 1
        try:
            im_path = os.path.join(database_path,images_path, split, image)
            img = cv2.imread(im_path)
            imfil = test_transformfilt(img)
            imfil = imfil.float().unsqueeze(dim=0).to(device)
            model3.eval()
            with torch.no_grad():
                pred = model3(imfil).argmax(dim=1)
                
            if pred == 1:                
                augmentations = val_transforms(image=img)
                img1 = augmentations["image"]
                       
                test_img = img1.float().to(device).unsqueeze(dim=0)
                        
                model2.eval()
                with torch.no_grad():
                    # pred1 = segm(model1(test_img))
                    pred2 = segm(model2(test_img)) 
                    
                predb2 = pred2 > 0.5
                predb2 = predb2.squeeze(dim=1)
                mask_pred = reverse_transform(predb2)
                # print(mask_pred.shape)
                mask_pred[mask_pred > 0.0] = 1.0
                y33 = mask_pred*255.0
                # print(y33.shape)
                y33 = y33.astype(np.uint8)
                y3 = cv2.dilate(y33,kernel,iterations = 1)
                # y3 = cv2.morphologyEx(y33, cv2.MORPH_GRADIENT, kernel)
                y3 = np.expand_dims(y3, 2)
                # y3 =  mask_pred+y3 
                # print(y3.shape)
                y3[y3 > 0.0] = 1.0
                # y3 = np.expand_dims(y3, 2)
                y3 = cv2.resize(y3, (512,512), interpolation = cv2.INTER_AREA)
                y3 = np.stack((y3, y3, y3), axis=2)
                
                lung_img = y3*img
                lung_img = cv2.cvtColor(lung_img, cv2.COLOR_BGR2GRAY)
                # lung_img = y3*cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
                # lung_img = clahe.apply(lung_img)
                equ = cv2.equalizeHist(lung_img)                   
            
                gray = cv2.cvtColor(cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                # gray2 = cv2.cvtColor(cv2.resize(lung_img, (224,224), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                imgs_lst.append(np.expand_dims(gray, 2)) 
                lung_lst.append(np.expand_dims(lung_img, 2)) 
                lung_lst2.append(np.expand_dims(equ, 2))
                
                    
        except:
            pass
        
    if len(imgs_lst) >3:
        tr_idx += 1
        labels.append(0)
        imgs_lst1 = np.concatenate(imgs_lst, axis=2)
        np.save(os.path.join(Train_save_path_Slice, str(tr_idx)+'.npy'), imgs_lst1) 
        
        lung_lst1 = np.concatenate(lung_lst, axis=2)
        np.save(os.path.join(Train_save_path_Slice, str(tr_idx)+'_lung.npy'), lung_lst1) 
        
        lung_lst2 = np.concatenate(lung_lst2, axis=2)
        np.save(os.path.join(Train_save_path_Slice, str(tr_idx)+'_lung2.npy'), lung_lst2) 

        
np.save(os.path.join(Train_save_path_Slice, 'labels.npy'), labels)       
        
     


