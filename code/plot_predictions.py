from utilities import *
from data import get_data, get_loaders
from model import get_model
from augmentations import *

import neptune
from accelerate import Accelerator, DistributedType
import torch
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns



def plot_predictions(CFG, fold, sample_size = 5):
    
    '''
    Display right and wrong predictions
    '''
    
    # tests
    assert isinstance(CFG, dict),  'CFG has to be a dict with parameters'
    assert isinstance(fold,  int), 'fold has to be an integer'
    assert sample_size > 0,        'sample_size has to be a positive int'

    # initialize accelerator
    accelerator = Accelerator(device_placement = True,
                              fp16             = CFG['use_fp16'],
                              split_batches    = False)
    accelerator.state.device = torch.device('cpu')

    # data sample
    oof          = pd.read_csv(CFG['out_path'] + 'oof.csv')
    oof          = oof.loc[oof['fold'] == fold].reset_index(drop = True)
    oof['error'] = np.abs(oof['pred'] - oof['target'])
    oof          = oof.sort_values(['error'], ascending = True)

    # split good and bad preds
    rights = oof.head(sample_size).reset_index(drop = True)
    wrongs = oof.tail(sample_size).reset_index(drop = True)

    # get data loaders
    _, right_loader = get_loaders(rights, rights, CFG, accelerator, silent = True)
    _, wrong_loader = get_loaders(wrongs, wrongs, CFG, accelerator, silent = True)

    # image grid
    fig = plt.figure(figsize = (20, 8))

    # right preds
    for batch_idx, (inputs, feats, labels) in enumerate(right_loader):
        for i in range(inputs.shape[0]):
            ax = fig.add_subplot(2, sample_size, i + 1, xticks = [], yticks = [])     
            plt.imshow(inputs[i].cpu().numpy().transpose(1, 2, 0))
            ax.set_title('{:.4f} [pred = {:.4f}]'.format(
                labels[i].numpy(), rights.iloc[i]['pred'], rights.iloc[i]['error']), color = 'green')

    # wrong preds
    for batch_idx, (inputs, feats, labels) in enumerate(wrong_loader):
        for i in range(inputs.shape[0]):
            ax = fig.add_subplot(2, sample_size, sample_size + i + 1, xticks = [], yticks = [])     
            plt.imshow(inputs[i].cpu().numpy().transpose(1, 2, 0))
            ax.set_title('{:.4f} [pred = {:.4f}]'.format(
                labels[i].numpy(), wrongs.iloc[i]['pred'], wrongs.iloc[i]['error']), color = 'red')

    # export
    plt.savefig(CFG['out_path'] + 'fig_errors.png')
    plt.show()