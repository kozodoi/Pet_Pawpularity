from utilities import *
from model import get_model
from train_fold import train_fold
from data import get_data
from plot_results import plot_results

import gc
import neptune
from accelerate import Accelerator, DistributedType
import torch
import pandas as pd
import numpy as np



def run_training(CFG, 
                 df, 
                 df_old = None, 
                 run    = None):
    
    '''
    Run cross-validation loop
    '''
    
    # tests
    assert isinstance(CFG, dict),        'CFG has to be a dict with parameters'
    assert isinstance(df, pd.DataFrame), 'df has to be a pandas dataframe'
    
    # placeholder
    oof_score = []

    # cross-validation
    for fold in range(CFG['num_folds']):
        
        
        # initialize accelerator
        accelerator = Accelerator(device_placement = True,
                                  fp16             = CFG['use_fp16'],
                                  split_batches    = False)
        if CFG['num_devices'] == 1 and CFG['device'] == 'GPU':
            accelerator.state.device = torch.device('cuda:{}'.format(CFG['device_index']))

        # feedback
        accelerator.print('-' * 55)
        accelerator.print('FOLD {:d}/{:d}'.format(fold + 1, CFG['num_folds']))    
        accelerator.print('-' * 55) 
        
        # load pretrained weights
        if CFG['pretrained'] == 'imagenet':
            pretrained = CFG['pretrained']
        elif CFG['pretrained'] == False:
            pretrained = CFG['pretrained']
        else:
            pretrained = CFG['pretrained'] + 'weights_fold{}.pth'.format(fold)
        
        # get model
        model = get_model(CFG        = CFG, 
                          pretrained = pretrained,
                          silent     = False)

        # get data
        df_trn, df_val = get_data(df          = df, 
                                  df_old      = df_old,
                                  fold        = fold, 
                                  CFG         = CFG, 
                                  accelerator = accelerator)  

        # run single fold
        trn_losses, val_losses, val_scores = train_fold(fold        = fold, 
                                                        df_trn      = df_trn,
                                                        df_val      = df_val, 
                                                        CFG         = CFG, 
                                                        model       = model, 
                                                        accelerator = accelerator,
                                                        run         = run)
        oof_score.append(np.min(val_scores))
        
        # feedback
        accelerator.print('-' * 55)
        accelerator.print('Best: score = {:.4f} (epoch {})'.format(
            np.min(val_scores), np.argmin(val_scores) + 1))
        accelerator.print('-' * 55)

        # plot loss dynamics
        if accelerator.is_local_main_process:
            plot_results(trn_losses, val_losses, val_scores, fold, CFG)

        # clear memory
        del accelerator
        gc.collect()
        
            
    # feedback
    print('')
    print('-' * 55)
    print('Mean OOF score = {:.4f}'.format(np.mean(oof_score)))
    print('-' * 55)
    if CFG['tracking']:
        run['oof_score'] = np.mean(oof_score)