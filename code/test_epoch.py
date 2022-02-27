from augmentations import get_tta_flips

import numpy as np
import timm
from timm.utils import *
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType



def test_epoch(loader, 
               model, 
               CFG,
               accelerator,
               num_tta = None):
    
    '''
    Run test epoch
    '''
    
    ##### PREPARATIONS
    
    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'
    
    # TTA options
    if num_tta is None:
        num_tta = CFG['num_tta']

    # switch regime
    model.eval()

    # placeholders
    SCORES = []
    
    # progress bar
    pbar = tqdm(range(len(loader)), disable = not accelerator.is_main_process)
    
    
    ##### INFERENCE LOOP
       
    # loop through batches
    with torch.no_grad():
        for batch_idx, (inputs, feats) in enumerate(loader):

            # preds placeholders
            scores = torch.zeros((inputs.shape[0], 1), device = accelerator.device)

            # compute predictions
            for tta_idx in range(num_tta): 
                if CFG['features']:
                    preds = model(get_tta_flips(inputs, tta_idx), feats)
                else: 
                    preds = model(get_tta_flips(inputs, tta_idx))
                if CFG['target_size'] != 1:
                    preds = preds[:, 0]
                    preds = preds[:, None]
                if CFG['loss_fn'] != 'MSE':
                    preds = torch.sigmoid(preds)
                scores += preds / num_tta

            # store predictions
            SCORES.append(accelerator.gather(scores).detach().cpu())
            
            # feedback
            pbar.update()

    # transform predictions
    return np.concatenate(SCORES)