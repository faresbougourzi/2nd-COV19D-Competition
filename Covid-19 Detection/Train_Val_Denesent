#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:02:50 2022

@author: bougourzi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
import numpy as np

from tqdm import tqdm

import os

from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2

img_size = 224
batch_size = 6
epochs = 60



############################
############################
from torch.utils import data
import albumentations as A

class Covid_loader_pt(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, path, transform=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.path = path
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        pth = self.path        
        labels = np.load(os.path.join(pth, 'labels.npy'))[ID]        
        X  = np.load(os.path.join(pth, str(ID) + '.npy'))
       
        if self.transform is not None:
            Xs = []
            data = self.transform(image=np.expand_dims(X[:,:,0],  2))
            Xs.append(data['image'])
            for i in range(X.shape[2]-1):
                image2_data = A.ReplayCompose.replay(data['replay'], image = np.expand_dims(X[:,:,i+1],  2))
                Xs.append(image2_data['image'])
       
        Xs = torch.stack(Xs).squeeze(dim = 1)
        Xs = F.interpolate(Xs.unsqueeze(dim = 0).unsqueeze(dim = 0), [64, 224, 224])

        return Xs, labels
############################

train_transforms = A.ReplayCompose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.VerticalFlip(p=0.1), A.Blur(blur_limit=(3, 3), p=0.2),
        A.MultiplicativeNoise(multiplier=1.5, p=0.2), 
        A.MultiplicativeNoise(multiplier=0.5, p=0.2),                  
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

val_transforms = A.ReplayCompose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[ 0.0],
            std=[ 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

torch.set_printoptions(linewidth=120)

tr_path = './NumpysSeg4/Train'
vl_path = './NumpysSeg4/Val'

tr_labels = np.load('./NumpysSeg4/Train/labels.npy')
vl_labels= np.load('./NumpysSeg4/Val/labels.npy')

tr_indxs= list(range(len(tr_labels)))
train_set = Covid_loader_pt(
        list_IDs = tr_indxs, 
        path = tr_path, 
        transform=train_transforms
)

vl_indxs= list(range(len(vl_labels)))
test_set = Covid_loader_pt(
        list_IDs = vl_indxs, 
        path = vl_path, 
        transform=val_transforms
)

#######
class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
    
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()     

####################################
class TwoInputsNet(nn.Module):
  def __init__(self):
    super(TwoInputsNet, self).__init__()
    self.encoder1 =  nn.Conv2d(64, 3, kernel_size = 3, stride = 1, padding = 1)
    # self.encoder2 =  nn.Conv2d(64, 3, kernel_size = 3, stride = 1, padding = 1)
    # self.encoder3 =  nn.Conv2d(64, 3, kernel_size = 3, stride = 1, padding = 1)
    # self.encoder4 =  nn.Conv2d(6, 3, kernel_size = 5, stride = 1, padding = 2)
    
    # self.encoder22 =  nn.Conv2d(9, 3, kernel_size = 3, stride = 1, padding = 1)
    # self.model = torchvision.models.resnext50_32x4d(pretrained=True)
    self.model = torchvision.models.densenet161(pretrained=True)
    # self.model = torchvision.models.resnet18(pretrained=True)

    self.model.classifier = nn.Linear(2208, 2) 
    self.act = nn.ReLU(inplace=True)
    

  def forward(self, input1):
    out1 = self.act(self.encoder1(input1))
    
    out = self.model(out1)
               
    return out
####################################

torch.set_grad_enabled(True)
torch.set_printoptions(linewidth=120)    


def MAE_distance(preds, labels):
    return torch.sum(torch.abs(preds - labels))

def Adaptive_loss(preds, labels, sigma):
    mse = (1+sigma)*((preds - labels)**2)
    mae = sigma + (torch.abs(preds - labels))
    return torch.mean(mse/mae)

def PC_mine(preds, labels):
    dem = np.sum((preds - np.mean(preds))*(labels - np.mean(labels)))
    mina = (np.sqrt(np.sum((preds - np.mean(preds))**2)))*(np.sqrt(np.sum((labels - np.mean(labels))**2)))
    return dem/mina 


device = torch.device("cuda:0") 
model = TwoInputsNet().to(device)     
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 16, shuffle = True)      
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 32)
criterion = FocalLoss()

epoch_count = []
Accracy_tr = []
Accracy_ts = []

Acc_best = -2
saving_models = "./Models 3d"
if not os.path.exists(saving_models):
    os.makedirs(saving_models)
name = './Models 3d/DensAlb88_focal10_best.pt'
################################## ############################################################    
for epoch in range(40):
    epoch_count.append(epoch)
    lr = 0.0001
    if epoch>15:
        lr = 0.00001  
    if epoch>30:
        lr = 0.000001        
        
    optimizer = optim.Adam(model.parameters(), lr = lr)
    train_loss = 0
    validation_loss = 0
    total_correct_tr = 0
    total_correct_val = 0
    total_correct_tr2 = 0
    total_correct_val2 = 0 
    
    label_f1tr = []
    pred_f1tr = []     
    
    for batch in tqdm(train_loader):        
        images1, labels = batch
        images1 = images1.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
        # images2 = images2.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
        # images3 = images3.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
        labels = labels.long().to(device)      
        torch.set_grad_enabled(True)
        model.train()
        preds= model(images1)
        loss = criterion(preds, labels)        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()            
             
        train_loss += loss.item()                   
        total_correct_tr +=  get_num_correct(preds, labels) 
        
        label_f1tr.extend(labels.cpu().numpy().tolist())
        pred_f1tr.extend(preds.argmax(dim=1).tolist())         

        del images1; del labels
        
    label_f1vl = []
    pred_f1vl = []        

    for batch in tqdm(validate_loader): 
        images1, labels = batch
        images1 = images1.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
        # images2 = images2.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
        # images3 = images3.squeeze(dim = 1).squeeze(dim = 1).float().to(device)        
        labels = labels.long().to(device)
        
        model.eval()
        with torch.no_grad():
            preds= model(images1)
            
        loss = criterion(preds, labels) 
                
        validation_loss += loss.item()                   
        total_correct_val +=  get_num_correct(preds, labels) 
        
        label_f1vl.extend(labels.cpu().numpy().tolist())
        pred_f1vl.extend(preds.argmax(dim=1).tolist())        
       

        del images1; del labels  
   
    
    print('Ep: ', epoch, 'AC_tr: ', total_correct_tr/len(train_set), 'AC_ts: ', total_correct_val/len(test_set), 'Loss_tr: ', train_loss/len(train_set))
    Acc_best2 = f1_score(label_f1vl, pred_f1vl , average='macro')
    
    print('MaF1_tr: ', f1_score(label_f1tr, pred_f1tr , average='macro'), 'MaF1_vl: ', \
          f1_score(label_f1vl, pred_f1vl , average='macro'))    
    
    Accracy_tr.append(total_correct_tr/len(train_set))
    Accracy_ts.append(Acc_best2)       
    
    if Acc_best2 >Acc_best: 
        Acc_best = Acc_best2
        torch.save(model.state_dict(), name)
                        
print(Acc_best) 

model.load_state_dict(torch.load(name))

###########################

label_f1vl = []
pred_f1vl = []


for batch in tqdm(validate_loader):
    images1, labels = batch
    images1 = images1.squeeze(dim=1).squeeze(dim=1).float().to(device)

    # images2 = images2.squeeze(dim=1).float().to(device)
    # images2 = images2.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
    # images3 = images3.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
    labels = labels.long().to(device)

    model.eval()
    with torch.no_grad():
        preds = model(images1)

    loss = criterion(preds, labels)
    # loss2 = criterion(preds2, labels2)
    # loss = loss1+loss2

    validation_loss += loss.item()
    total_correct_val += get_num_correct(preds, labels)
    # total_correct_val2 += get_num_correct(preds2, labels2)

    label_f1vl.extend(labels.cpu().numpy().tolist())
    pred_f1vl.extend(preds.argmax(dim=1).tolist())

    del images1
    del labels

print(Acc_best)

print(confusion_matrix(label_f1vl, pred_f1vl))
