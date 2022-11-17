import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

print("Torch version:", torch.__version__)

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import scipy.io as sio
from sklearn import preprocessing
import sys
from pathlib import Path
import pickle
import copy
import glob
from ..dataloader import *

dataset = 'CUB' # SUN or AWA2
backbone = 'vit_huge_patch14_224_in21k'
steps = 200
LR=0.0001

if dataset == "CUB":
    num_classes = 150
elif dataset == "SUN":
    num_classes = 645
elif dataset == "AWA2":
    num_classes = 40

model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
model = model.cuda()
config = resolve_data_config({}, model=model)
preprocess = create_transform(**config)


dataset_unseen = DATA(loader_type='test_unseen_loc', transform=preprocess)
dataset_loader_unseen = torch.utils.data.DataLoader(dataset_unseen,
                                            batch_size=32, shuffle=False,
                                            num_workers=4)

dataset_seen = DATA(loader_type='test_seen_loc', transform=preprocess)
dataset_loader_seen = torch.utils.data.DataLoader(dataset_seen,
                                            batch_size=32, shuffle=False,
                                            num_workers=4)

dataset_train = DATA(loader_type='trainval_loc', transform=preprocess)
dataset_loader_training = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=100, shuffle=True,
                                            num_workers=4)

all_losses = []
all_seen_acc = []
all_unseen_acc = []

model.train()
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
model_optimizer = torch.optim.Adam(trainable_parameters, lr=1e-3,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001) 
loss_img = nn.CrossEntropyLoss()
# loss_txt = nn.CrossEntropyLoss()

for step in range(steps):    
    
    if step == 100:
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        model_optimizer = torch.optim.Adam(trainable_parameters, lr=1e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001) 
        torch.save({'model': model.state_dict(), 'epoch': 100,  'lr': 5e-8, 'betas': (0.9,0.98), 'eps': 1e-6, 'weight_decay': 0.001}, 
                f'{backbone}_100epochs_{dataset}.ckpt')
        
    model.train()
    log_total_loss = 0
    for i, (images, target, _, _, _) in enumerate(dataset_loader_training):
        model_optimizer.zero_grad()
        
        images = images.cuda()
        target = target.type(torch.LongTensor).squeeze().cuda()

        logits_per_image = model(images)

        total_loss = loss_img(logits_per_image, target)
    
        # print ('loss: ', total_loss)
        log_total_loss += total_loss.item()
        model_optimizer.zero_grad()
        total_loss.backward()
        
        model_optimizer.step()
        
    print ('step: ', step, ' - loss: ', log_total_loss / i)
    all_losses.append(log_total_loss / i)
    
    if step % 10 == 0:
        model.eval()

        all_unseen_targets = []
        all_unseen_predictions = []
        actual_all_unseen_targets = []

        for i, (images, target, _) in enumerate(dataset_loader_seen):

            images = images.cuda()
            target = target.squeeze().cuda()
            actual_all_unseen_targets.extend(target.cpu().numpy())

            outputs = model(images)
            probs = outputs.softmax(dim=-1) #.detach().cpu().numpy()
            top_probs, top_labels = probs.cpu().topk(1, dim=-1)
            all_unseen_predictions.extend(top_labels.squeeze().detach().cpu().numpy().squeeze())

        target_classes = np.unique(actual_all_unseen_targets) # np.unique(test_unseen_label)
        test_label = actual_all_unseen_targets # all_unseen_targets
        predicted_label = all_unseen_predictions

        per_class_accuracies_unseen = Variable(torch.zeros(target_classes.shape[0]).float().to('cuda')).detach()
        predicted_label = np.array(predicted_label)
        predicted_label = torch.tensor(predicted_label).to('cuda')
        test_label = np.array(test_label)
        test_label = torch.tensor(np.array(test_label)).to('cuda')

        for i in range(target_classes.shape[0]):
            is_class = test_label==target_classes[i]
            equal = (predicted_label[is_class]==test_label[is_class]).sum()
            divisor = float(is_class.sum())
            per_class_accuracies_unseen[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum(),float(is_class.sum()))        

        seen_acc = per_class_accuracies_unseen.mean().item()
        print ('seen_acc:', seen_acc)
        all_seen_acc.append(seen_acc)

torch.save({'model': model.state_dict(), 'epoch': 100,  'lr': 5e-8, 'betas': (0.9,0.98), 'eps': 1e-6, 'weight_decay': 0.001}, 
                f'{backbone}_100epochs_{dataset}.ckpt')

