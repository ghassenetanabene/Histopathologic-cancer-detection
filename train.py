import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import cv2
import random
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import albumentations
import albumentations.pytorch.transforms as AT

from dataset import CancerDataset
from models.model import get_model
from utils.utils import plant_seed

def main(args):
    """
    Main function that runs the training loop
    
    Paramter:
    --------
    args: Hyperparamters
    """
    
    # Labels of Training set
    labels = pd.read_csv(os.path.join(args.path,'train_labels.csv'))
    
    # Splitting the dataset into train and validation set
    tr, val = train_test_split(labels.label, stratify=labels.label, test_size=0.101, random_state=args.seed)
    
    img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}
    
    # Data Augmentation for training
    data_transforms = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.RandomRotate90(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Flip(p=0.5),
        albumentations.OneOf([
            albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(), albumentations.IAAEmboss(), 
            albumentations.RandomBrightness(), albumentations.RandomContrast(),
            albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5), 
        albumentations.HueSaturationValue(p=0.5), 
        albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        AT.ToTensor()
    ])

    # Data Augmentation for validation
    data_transforms_val = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        AT.ToTensor()
        ])
    
    # Creating the dataset
    dataset = CancerDataset(datafolder=os.path.join(args.path, 'train'), datatype='train',
                            transform=data_transforms, labels_dict=img_class_dict)
    val_set = CancerDataset(datafolder=os.path.join(args.path, 'train'), datatype='train',
                            transform=data_transforms_val, labels_dict=img_class_dict)
    print ('='*40)
    print ('Dataset Created')
    print ('='*40)
    
    # Sampling the dataset to prevent overfitting
    train_sampler = SubsetRandomSampler(list(tr.index)) 
    valid_sampler = SubsetRandomSampler(list(val.index))
    
    # Loading the hparams
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    # Creating the dataloader
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                              sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(val_set, batch_size=batch_size,
                              sampler=valid_sampler, num_workers=num_workers)
    
    print ('='*40)
    print ('Dataset Loaded')
    print ('='*40)
    
    # Creating the model
    model_name = args.model_name
    model = get_model(model_name, pretrained=args.pretrained)
    print ('='*40)
    print ('Model Created')
    print ('='*40)
    
    # Setting up the loss function
    criterion = nn.BCEWithLogitsLoss()
    # Setting the optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    # Setting the scheduler
    scheduler = StepLR(optimizer, step_size = args.step_size, gamma=args.gamma)
    scheduler.step()
    
    # Tensorboard for logging the experiments
    writer = SummaryWriter(log_dir= 'logs')
    
    # For saving the best Model based on validation AUC score
    val_auc_max = 0
    
    # Waiting steps till val auc does not increase
    patience = args.patience
    
    # Current number of tests, where validation loss didn't increase
    p = 0
    # Whether training should be stopped
    stop = args.stop

    # Number of epochs to train the model
    n_epochs = args.epochs
    
    print ('='*40)
    print ('Starting Training')
    print ('='*40)
    
    for epoch in range(1, n_epochs+1):
    
        if stop:
            print("Training stop.")
            break
            
        print(time.ctime(), 'Epoch:', epoch)

        train_loss = []
        train_auc = []
            
        for tr_batch_i, (data, target) in enumerate(train_loader):
            
            model.train()
            
            if torch.cuda.is_available():
                model.cuda()
                data = data.cuda()
                target = target.cuda()
                criterion = criterion.cuda()

            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output[:,0], target.float())
            train_loss.append(loss.item())
            
            a = target.data.cpu().numpy()
            try:
                b = output[:,0].detach().cpu().numpy()
                train_auc.append(roc_auc_score(a, b))
            except:
                pass

            loss.backward()
            optimizer.step()
            
            # Evaluating the model during training time
            if (tr_batch_i+1) % args.eval_every == 0:    
                model.eval()
                val_loss = []
                val_auc = []
                
                for val_batch_i, (data, target) in enumerate(valid_loader):
                    if torch.cuda.is_available():
                        model = model.cuda()
                        data = data.cuda()
                        target = target.cuda()
                        criterion = criterion.cuda()
                        
                    output = model(data)

                    loss = criterion(output[:,0], target.float())

                    val_loss.append(loss.item()) 
                    a = target.data.cpu().numpy()
                    try:
                        b = output[:,0].detach().cpu().numpy()
                        val_auc.append(roc_auc_score(a, b))
                    except:
                        pass
                
                # Logging the Losses and Accuracy
                writer.add_scalar('train/loss', np.mean(train_loss), epoch * len(train_loader) + tr_batch_i)
                writer.add_scalar('val/loss', np.mean(val_loss), epoch * len(train_loader) + tr_batch_i)
                writer.add_scalar('train/acc', np.mean(train_auc), epoch * len(train_loader) + tr_batch_i)
                writer.add_scalar('val/acc', np.mean(val_auc), epoch * len(train_loader) + tr_batch_i)
                
                print('[Epoch] %d, [batches]:%d, [Train loss]: %.4f, [Val loss]: %.4f.'%\
                      (epoch, tr_batch_i, np.mean(train_loss), np.mean(val_loss)) 
                    +'  [Train auc]: %.4f, [Val auc]: %.4f'% (np.mean(train_auc), np.mean(val_auc)))
                
                train_loss = []
                train_auc = []
                valid_auc = np.mean(val_auc)
                
                if valid_auc > val_auc_max:
                    print('Validation auc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        val_auc_max,
                        valid_auc)
                        )
                    
                    # saving the model
                    if not os.path.exists('saved_models'):
                        os.makedirs('saved_models')
                    save_name = os.path.join('saved_models', 'model.pt')
                    torch.save(model.state_dict(), save_name)
                    print ('='*40)
                    print (f'Model Saved at Epoch : {epoch}')
                    print ('='*40)
                    
                    val_auc_max = valid_auc
                    p = 0
                    
                else:
                    p += 1
                    if p > patience:
                        print('Early stop training')
                        stop = True
                        break  
                     
                scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type= str, default = '../input/histopathologic-cancer-detection')
    parser.add_argument('--model_name', type = str, default='cbam_resnet50')
    parser.add_argument('--pretrained', default=True, action='store_false')
    parser.add_argument('--step_size', type= int, default= 5)
    parser.add_argument('--gamma', type = float, default = 0.2)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--num_workers', type = int, default= 4)
    parser.add_argument('--stop', default=False, action= 'store_true')
    parser.add_argument('--patience', default= 25, type = int)
    parser.add_argument('--epochs', type = int, default=5)
    parser.add_argument('--lr', type= float, default=4e-4)
    parser.add_argument('--eval_every', type= int, default=600)
    parser.add_argument('--seed', type= int, default=323)
    args = parser.parse_args()
    print(args)
    
    plant_seed(args.seed)
    print ('seeded')
    
    main(args)
