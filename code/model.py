####### MODEL PREP

from utilities import *
import timm
import torch
import torch.nn as nn
import gc

import sys
sys.path.append('../convnext/code')
#sys.path.append('../input/convnext-models-code/code')
import models.convnext
import models.convnext_isotropic
import utils

def get_model(CFG, 
              pretrained = None, 
              silent     = False):
    
    '''
    Instantiate the model
    '''
    
    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'
    
    # pretrained weights
    if pretrained is None:
        pretrained = CFG['pretrained']
            
    # model class
    class Model(nn.Module):

        def __init__(self, CFG, pretrained):
            super().__init__()
            
            # body
            if 'convnext' not in CFG['backbone']:
                self.backbone = timm.create_model(model_name  = CFG['backbone'], 
                                                  pretrained  = True if pretrained == 'imagenet' else False,
                                                  in_chans    = 3,
                                                  num_classes = 0)

            # body for convnext
            if 'convnext' in CFG['backbone']:
                
                if 'base' in CFG['backbone']:
                    model = timm.create_model(model_name  = 'convnext_base', 
                                              pretrained  = False,
                                              in_chans    = 3,
                                              num_classes = 128)
                elif 'large' in CFG['backbone']:
                    model = timm.create_model(model_name  = 'convnext_large', 
                                              pretrained  = False,
                                              in_chans    = 3,
                                              num_classes = 128)
                    
                if CFG['environment'] == 'local':
                    checkpoint = torch.load('../convnext/' + CFG['backbone'] + '.pth', map_location = 'cpu')
                elif CFG['environment'] == 'kaggle':
                    checkpoint = torch.load('../input/convnext-models-code/ImageNet22K/' + CFG['backbone'] + '.pth', map_location = 'cpu')
                print('-- loading convnext weights...')
                
                checkpoint_model = checkpoint['model']
                state_dict       = model.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        del checkpoint_model[k]
                utils.load_state_dict(model, checkpoint_model, prefix = '')
                self.backbone = model

            # head
            layers = []
            if CFG['batchnorm']:
                layers.append(nn.LazyBatchNorm1d())
            if CFG['dropout'] > 0:
                layers.append(nn.Dropout(CFG['dropout']))
            layers.append(nn.LazyLinear(CFG['target_size']))
            if CFG['loss_fn'] == 'MSE':
                layers.append(nn.Sigmoid())
            self.fc = nn.Sequential(*layers)
            
                
        def forward(self, image, features = None):
            x = self.backbone(image)
            if features is not None:
                x = torch.cat([x, features], dim = 1)
            x = self.fc(x)
            return x
                      
    # get model
    model = Model(CFG, pretrained)
            
    # load pre-trained weights
    if pretrained:
        if pretrained != 'imagenet':
            model.load_state_dict(torch.load(pretrained, map_location = torch.device('cpu')))
            if not silent:
                print('-- loaded weights:', pretrained)
        else:
            if not silent:
                print('-- loaded imagenet weights')
            
    return model