import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import cv2

import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import time 
import random

import argparse

import albumentations
import albumentations.pytorch.transforms as AT

from dataset import CancerDataset
from models.model import get_model
from utils.utils import plant_seed, submission, predict

def main(args):
    """
    Main function to do Inference on the trained model
    """
    
    # Various Data Augmentations to perform during test time 
    data_transforms_test = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        AT.ToTensor()
    ])
    
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
    
    data_transforms_tta0 = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.RandomRotate90(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Flip(p=0.5),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        AT.ToTensor()
        ])

    data_transforms_tta1 = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.RandomRotate90(p=1),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        AT.ToTensor()
        ])

    data_transforms_tta2 = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Transpose(p=1),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        AT.ToTensor()
        ])

    data_transforms_tta3 = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Flip(p=1),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        AT.ToTensor()
        ])
    
    # Initializing the various hparams
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_tta = args.num_tta
    use_tta = args.use_tta
    
    # Load the Model
    model_name = args.model_name
    model = get_model(model_name, pretrained=True)
    print ('='*40)
    print ('Model Initialized')
    print ('='*40)
    
    # Load the Pretrained model on Cancer dataset
    saved_dict = torch.load(os.path.join(args.model_path, args.ckpt_name))
    model.load_state_dict(saved_dict)
    print ('='*40)
    print ('Pretrained Model Loaded')
    print ('='*40)
    
    # Test Path
    test_path = os.path.join(args.path, 'test')
    
    if not use_tta:
        
        # Loading the Dataset
        test_set = CancerDataset(datafolder=test_path, datatype='test',
                                transform=data_transforms_test)
        
        test_loader = DataLoader(test_set, batch_size=batch_size, 
                                num_workers=num_workers)
        
        # Prediction if Not TTA
        preds = predict(model, test_loader, use_tta, num_tta)
        test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})
        test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])
        
        # Create a submission file
        submission(args.path, test_preds)
    
    else:
        # Perform various TTA
        for i in range(num_tta):
            if i==0:
                test_set = CancerDataset(datafolder=test_path, datatype='test', transform=data_transforms_test)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
            elif i==1:
                test_set = CancerDataset(datafolder=test_path, datatype='test', transform=data_transforms_tta1)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
            elif i==2:
                test_set = CancerDataset(datafolder=test_path, datatype='test', transform=data_transforms_tta2)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
            elif i==3:
                test_set = CancerDataset(datafolder=test_path, datatype='test', transform=data_transforms_tta3)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
            elif i<8:
                test_set = CancerDataset(datafolder=test_path, datatype='test', transform=data_transforms_tta0)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
            else:
                test_set = CancerDataset(datafolder=test_path, datatype='test', transform=data_transforms)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

            # Make prediction
            preds = predict(model, test_loader, use_tta, num_tta)
            
            if i==0:
                test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})
                test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])
            else:
                test_preds['preds']+=np.array(preds)
        
        # Create Submission
        submission(args.path, test_preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type= str, default = '../input/histopathologic-cancer-detection')
    parser.add_argument('--model_name', type = str, default='cbam_resnet50')
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--num_workers', type = int, default= 4)
    parser.add_argument('--seed', type= int, default=323)
    parser.add_argument('--use_tta', default= False, action = 'store_true')
    parser.add_argument('--num_tta', default = 32, type = int)
    parser.add_argument('--model_path', default='saved_models/', type = str)
    parser.add_argument('--ckpt_name', default= 'model.pt', type = str)
    
    args = parser.parse_args()
    print(args)
    
    plant_seed(args.seed)
    print ('seeded')
    
    main(args)