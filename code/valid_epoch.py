from augmentations import get_tta_flips

import numpy as np
import timm
from timm.utils import *
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType



def valid_epoch(loader, 
                model, 
                criterion, 
                CFG, 
                accelerator,
                num_tta = None):
        
    '''
    Validation epoch
    '''
    
    ##### PREPARATIONS
    
    # TTA options
    if num_tta is None:
        num_tta = CFG['num_tta']

    # switch regime
    model.eval()

    # running loss
    val_loss = AverageMeter()

    # placeholders
    SCORES = []
    LABELS = []
    
    # progress bar
    pbar = tqdm(range(len(loader)), disable = not accelerator.is_main_process)
                
                
    ##### INFERENCE LOOP
       
    # loop through batches
    with torch.no_grad():
        for batch_idx, (inputs, feats, labels) in enumerate(loader):

            # preds placeholders
            scores = torch.zeros((inputs.shape[0], CFG['target_size']), device = accelerator.device)

            # compute predictions
            for tta_idx in range(num_tta): 
                if CFG['features']:
                    preds = model(get_tta_flips(inputs, tta_idx), feats)
                else: 
                    preds = model(get_tta_flips(inputs, tta_idx))
                if CFG['loss_fn'] != 'MSE':
                    preds = torch.sigmoid(preds)
                scores += preds / num_tta

            # compute loss
            if CFG['target_size'] == 1:
                loss = criterion(scores.view(-1), labels)
            else:
                loss = criterion(scores, labels)
            val_loss.update(loss.item(), inputs.size(0))

            # store predictions
            SCORES.append(accelerator.gather(scores).detach().cpu())
            LABELS.append(accelerator.gather(labels).detach().cpu())
            
            # feedback
            pbar.update()

    # transform predictions
    return val_loss.sum, np.concatenate(SCORES), np.concatenate(LABELS)