import torch.optim as optim
from adamp import AdamP
from madgrad import MADGRAD



def get_optimizer(CFG, model):
    
    '''
    Get optimizer
    '''
              
    ##### PREPARATIONS
    
    # params
    wd = CFG['decay']
    lr = CFG['lr']
 
    # params in head
    parameters = [{'params':       [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad],
                   'weight_decay': wd,
                   'lr':           lr}]

    # list of backbone layers
    backbone_layers = [n for n, p in model.named_parameters() if 'backbone' in n]
    backbone_layers.reverse()

    # params in backbone
    for layer in backbone_layers:
        parameters += [{'params':       [p for n, p in model.named_parameters() if layer in n and p.requires_grad],
                        'weight_decay': wd,
                        'lr':           lr}]
        lr *= CFG['lr_decay'] 
        
        
    ##### OPTIMIZERS
    
    if CFG['optim'] == 'Adam':
        optimizer = optim.Adam(parameters)
    elif CFG['optim'] == 'AdamW':
        optimizer = optim.AdamW(parameters)
    elif CFG['optim'] == 'AdamP':
        optimizer = AdamP(parameters,)
    elif CFG['optim'] == 'madgrad':
        optimizer = MADGRAD(parameters) 


    return optimizer