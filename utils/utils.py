import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from scipy.special import expit #for sigmoid function

def plant_seed(seed):
    """
    Seeds everything for experiments iteration
    
    Parameter: 
    ---------
    seed : int
        Takes the seed
    """
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def submission(path, test_preds):
    """
    Creates a submission file to be uploaded on Kaggle.
    
    Parameter:
    ---------
    path: Path of sample submission
    test_preds: Array of predictions from saved model
    """
    sub = pd.read_csv(os.path.join(path,'sample_submission.csv'))
    sub = pd.merge(sub, test_preds, left_on='id', right_on='imgs')
    sub = sub[['id', 'preds']]
    sub.columns = ['id', 'label']

    sub.to_csv('submission.csv',index = False)
    print ('='*40)
    print ('Submission Created')
    print ('='*40)
  
def predict(model, test_loader, use_tta, num_tta):
    """
    Helper function for making the prediction
    
    Parameter:
    ---------
    model : Saved model
    test_loader : Dataloader for the test dataset
    use_tta : Boolean, to use TTA (Test Time Augmentation) during testing time
    num_tta : int, Number of TTA to apply
    
    Returns:
    -------
    preds : Returns the predictions on test set
    """
    preds = []
    model.eval()
    sigmoid = lambda x: expit(x)
    
    print ('-'* 40)
    print ('Predicting')
    for _, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            model = model.cuda()
            data = data.cuda()
            target = target.cuda()
        
        output = model(data).detach()
        pr = output[:,0].cpu().numpy()
        
        for i in pr:
            if use_tta:
                preds.append(sigmoid(i)/num_tta)
            else:
                preds.append(i)
            
    return preds
