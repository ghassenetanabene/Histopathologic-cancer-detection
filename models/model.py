__all__ = ['CbamResNet', 'cbam_resnet18', 'cbam_resnet34', 
           'cbam_resnet50', 'cbam_resnet101', 'cbam_resnet152']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .cbamresnet import (cbam_resnet18,
                        cbam_resnet34,
                        cbam_resnet50,
                        cbam_resnet101,
                        cbam_resnet152
                        )

def get_model(model_name, pretrained = True):
    """
    Helper function for creating the CBAM Model
    Adds new layers for transfer learning
    
    Parameter:
    ---------
    model_name: str
        Specifies the Model name to use
    pretrained : Boolean
        Use ImageNet pretrained or not
    
    Returns:
    -------
    model : returns the newly created model
    """
    if '18' in model_name:
        model = cbam_resnet18(pretrained = pretrained)
    elif '34' in model_name:
        model = cbam_resnet34(pretrained = pretrained)
    elif '50' in model_name:
        model = cbam_resnet50(pretrained = pretrained)
    elif '101' in model_name:
        model = cbam_resnet101(pretrained = pretrained)
    else:
        model = cbam_resnet152(pretrained = pretrained)
    
    # Adds New layers for transfer learning
    model.avg_pool  = nn.AdaptiveAvgPool2d((1, 1))
    model.last_linear = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(in_features=2048, out_features=512, bias=True),
        nn.SELU(),
        nn.Dropout(0.8),
        nn.Linear(in_features=512, out_features=1, bias=True)
        )
    
    return model
    
    

