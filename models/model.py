__all__ = ['CbamResNet', 'cbam_resnet18', 'cbam_resnet34', 
           'cbam_resnet50', 'cbam_resnet101', 'cbam_resnet152']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet50
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
    if 'cbam_resnet18' is model_name:
        model = cbam_resnet18(pretrained = pretrained)
    elif 'cbam_resnet34' is model_name:
        model = cbam_resnet34(pretrained = pretrained)
    elif 'cbam_resnet50' is model_name:
        model = cbam_resnet50(pretrained = pretrained)
    elif 'cbam_resnet101' is model_name:
        model = cbam_resnet101(pretrained = pretrained)
    elif 'cbam_resnet152' is model_name:
        model = cbam_resnet152(pretrained = pretrained)
    elif 'resnet50' == model_name:
        model = resnet50(pretrained = pretrained)
    
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
    
    

