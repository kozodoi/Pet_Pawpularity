import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import numpy as np
import pandas as pd

from utilities import *
from augmentations import get_augs



class ImageData(Dataset):
    
    '''
    Image dataset class
    '''
    
    def __init__(self, 
                 df, 
                 target_size = 1,
                 labeled     = True,
                 transform   = None):
        self.df          = df
        self.target_size = target_size
        self.labeled     = labeled
        self.transform   = transform
        self.features    = ['Subject Focus', 'Eyes', 'Face', 'Near', 
                            'Action', 'Accessory', 'Group', 'Collage', 
                            'Human', 'Occlusion', 'Info', 'Blur']
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        # import image
        path  = self.df.loc[idx, 'file_path']
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # import meta features
        feats = torch.tensor(self.df.iloc[idx][self.features]).float()
            
        # augmentations
        if self.transform is not None:
            image = self.transform(image = image)['image']
                    
        # output
        if self.labeled:
            if self.target_size == 1:
                label = torch.tensor(self.df.iloc[idx]['target'], dtype = torch.float16)
            elif self.target_size == 13:
                label = torch.tensor(self.df.iloc[idx][['target'] + self.features]).float()
            return image, feats, label
        else:
            return image, feats     

    

def get_data(df, fold, CFG, accelerator, df_old = None, silent = False, debug = None):
    
    '''
    Get training and validation data
    '''
    
    # tests
    assert isinstance(df, pd.DataFrame), 'df has to be a pandas dataframe'
    assert isinstance(fold,  int),       'fold has to be an integer'
    assert isinstance(CFG, dict),        'CFG has to be a dict with parameters'

    # load splits
    df_train = df.loc[df.fold != fold].reset_index(drop = True)
    df_valid = df.loc[df.fold == fold].reset_index(drop = True)
    if not silent:
        accelerator.print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)))
        
    # remove duplicates
    if CFG['drop_dupl']:
        if CFG['drop_dupl'] == 0.95:
            train_duplicates = find_duplicates(train_ids = list(df_train['Id']), 
                                              valid_ids = list(df_valid['Id']))
        if CFG['drop_dupl'] == 0.85: 
            train_duplicates = ['bf8501acaeeedc2a421bac3d9af58bb7', '72b33c9c368d86648b756143ab19baeb', '86547ebc49a622ef1ebb814f3fe93327', '9b3267c1652691240d78b7b3d072baf3', '6ae42b731c00756ddd291fa615c822a1', '9aa6a6702b67912a8d99e1afa5c42b9a', '87c6a8f85af93b84594a36f8ffd5d6b8', '506df6a6470c293c95c82ba61e87fc9a', '2b737750362ef6b31068c4a4194909ed', '592fdd2282ab0dad710d51555521451e', '8f3f1d62e9020538d0069586a216411c', '61b2e3c70eb4d9e9d198cae78b3f80d8', '9262ff92924c32b085360bcce65a2ca1', 'e359704524fa26d6a3dcd8bfeeaedd2e', '38426ba3cbf5484555f2b5e9504a6b03', '4b618cd51feea66ef7e30489f8effa36', '1a00622c9108fb4ddcd716ce6a20b7b5', '4900e463696cc862724c3aa96b5d7487', '5a642ecc14e9c57a05b8e010414011f2', '36a989f70c5090f27fa6f36961a35fa9', '8f3f1d62e9020538d0069586a216411c', '24c31962708293f176b0839c11aa95cd', '02b65b9132cd726756a300a4dd3ecd4a', '8ffde3ae7ab3726cff7ca28697687a42', '869326d56a88de6e4983d90883c0e09c', '1a00622c9108fb4ddcd716ce6a20b7b5', 'abc3bc25190cf419862c6c7e10f14e77', 'b190f25b33bd52a8aae8fd81bd069888', 'fe4854282aebafd366a9fbc36364ef60', '9a0238499efb15551f06ad583a6fa951', 'cdce90fba529efbd06a374cbfe541fdf', '1f03a452abe5f323d64d8438f187482b', '56237f3775069bdf055442f0a8e25676', '8f1bca10c606cba1c9741ad1bb26cc36', 'a883c48295d89d73e5a2b2d41cfecc56', '988b31dd48a1bc867dbc9e14d21b05f6', 'ec2e8a8c7eb931aab97e4c219c9d3fcd', '2003690eccabb930ee712d7387466765', 'f15868f7fc13687acde228a81334df8d', '78a02b3cb6ed38b2772215c0c0a7f78e', 'dbc47155644aeb3edd1bd39dba9b6953', '8a12d4f4f068b697e98eec4697e26c7a', '481ec365ebb75fe0382e0061b9787657', '8f12f92eaa9b021ba7fe9bfce7163c8a', '16d8e12207ede187e65ab45d7def117b', 'dd042410dc7f02e648162d7764b50900', '5790c1c4cf08bf966b84b1a39e419b63', '54563ff51aa70ea8c6a9325c15f55399', 'a8f044478dba8040cc410e3ec7514da1', '3877f2981e502fe1812af38d4f511fd2', '5ef7ba98fc97917aec56ded5d5c2b099', '1a00622c9108fb4ddcd716ce6a20b7b5', 'fe47539e989df047507eaa60a16bc3fd', '94c823294d542af6e660423f0348bf31', 'b65eaed98a65afc9773a1454217799c8', '0422cd506773b78a6f19416c98952407', '05010a08dc04beffa845696c357676df', 'c5e453e9db92cb9ab31d1b73c31fbea1', 'cd909abf8f425d7e646eebe4d3bf4769', 'b49ad3aac4296376d7520445a27726de', 'ac8823ae3ef0da44d00c9f7fab5ed3ea', 'b148cbea87c3dcc65a05b15f78910715', '01430d6ae02e79774b651175edd40842', '54b3fcd098cf36945de9d06e24bbeddf', '441a857042cb86eff990a8db8f40ff83', 'a9513f7f0c93e179b87c01be847b3e4c', '1059231cf2948216fcc2ac6afb4f8db8', '9f5a457ce7e22eecd0992f4ea17b6107', '365d8c50030e64ac8ebcb4d738876c2c', '13d215b4c71c3dc603cd13fc3ec80181', '1feb99c2a4cac3f3c4f8a4510421d6f5', '5143a57743d830172f351751094c92fb', 'fea5968d4979d8d98d964c31e0c7fe66', '776d92567e4d173093527335741f49c6', '9cfeb5d1dd7191a7925e6da0ef6f794a']
        len_before = len(df_train)
        df_train = df_train.loc[~df_train.Id.isin(train_duplicates)].reset_index(drop = True)
        if not silent:
            accelerator.print('- removing {} duplicates from training data...'.format(len_before - len(df_train)))
            accelerator.print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)))
            
    # add external data
    if CFG['old_data']:
        df_old_sample = df_old.sample(frac = CFG['old_data'], replace = False, random_state = CFG['seed'] + fold) 
        df_train      = pd.concat([df_train, df_old_sample], axis = 0).reset_index(drop = True)
        if not silent:
            accelerator.print('- appending old labeled data to train...')
            accelerator.print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)))
            
    # subset for debug mode
    if debug is None:
        debug = CFG['debug']
    if debug:
        df_train = df_train.sample(CFG['batch_size'] * 10, random_state = CFG['seed']).reset_index(drop = True)
        df_valid = df_valid.sample(CFG['batch_size'] * 10, random_state = CFG['seed']).reset_index(drop = True)
        accelerator.print('- subsetting data for debug mode...')
        accelerator.print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)))
    
    return df_train, df_valid



def get_loaders(df_train, df_valid, CFG, accelerator, labeled = True, shuffle = True, silent = False):
    
    '''
    Get training and validation dataloaders
    '''
    
    # tests
    assert isinstance(df_train, pd.DataFrame), 'df_train has to be a pandas dataframe'
    assert isinstance(df_valid, pd.DataFrame), 'df_valid has to be a pandas dataframe'
    assert isinstance(CFG, dict),              'CFG has to be a dict with parameters'

    ##### DATASETS
        
    # augmentations
    train_augs, valid_augs = get_augs(CFG        = CFG, 
                                      image_size = CFG['image_size'], 
                                      p_aug      = CFG['p_aug'])
    
    # datasets
    train_dataset = ImageData(df          = df_train, 
                              target_size = CFG['target_size'],
                              transform   = train_augs,
                              labeled     = labeled)
    valid_dataset = ImageData(df          = df_valid, 
                              target_size = CFG['target_size'],
                              transform   = valid_augs,
                              labeled     = labeled)

        
    ##### DATA LOADERS
    
    # data loaders
    train_loader = DataLoader(dataset        = train_dataset, 
                              batch_size     = CFG['batch_size'], 
                              shuffle          = shuffle,
                              num_workers    = CFG['cpu_workers'],
                              drop_last      = False, 
                              worker_init_fn = worker_init_fn,
                              pin_memory     = False)
    valid_loader = DataLoader(dataset     = valid_dataset, 
                              batch_size  = CFG['valid_batch_size'], 
                              shuffle       = False,
                              num_workers = CFG['cpu_workers'],
                              drop_last   = False,
                              pin_memory  = False)
    
    # feedback
    if not silent:
        accelerator.print('- image size: {}, p(augment): {}'.format(CFG['image_size'], CFG['p_aug']))
        accelerator.print('-' * 55)
    
    return train_loader, valid_loader


def find_duplicates(train_ids, valid_ids):
    
    '''
    Return list of train IDs of the duplicates
    '''
        
    # list of duplicate pairs
    # adapted from https://www.kaggle.com/schulta/petfinder-identify-duplicates-and-share-findings
    duplicate_list = {('01430d6ae02e79774b651175edd40842', '6dc1ae625a3bfb50571efedc0afc297c'),
                      ('03d82e64d1b4d99f457259f03ebe604d', 'dbc47155644aeb3edd1bd39dba9b6953'),
                      ('0422cd506773b78a6f19416c98952407', '0b04f9560a1f429b7c48e049bcaffcca'),
                      ('08440f8c2c040cf2941687de6dc5462f', 'bf8501acaeeedc2a421bac3d9af58bb7'),
                      ('0c4d454d8f09c90c655bd0e2af6eb2e5', 'fe47539e989df047507eaa60a16bc3fd'),
                      ('1059231cf2948216fcc2ac6afb4f8db8', 'bca6811ee0a78bdcc41b659624608125'),
                      ('13d215b4c71c3dc603cd13fc3ec80181', '373c763f5218610e9b3f82b12ada8ae5'),
                      ('1feb99c2a4cac3f3c4f8a4510421d6f5', '264845a4236bc9b95123dde3fb809a88'),
                      ('221b2b852e65fe407ad5fd2c8e9965ef', '94c823294d542af6e660423f0348bf31'),
                      ('2b737750362ef6b31068c4a4194909ed', '41c85c2c974cc15ca77f5ababb652f84'),
                      ('36b4d2c3227812c555b14565eb65508a', 'bce09001893018dd48c19a0fd6a5b065'),
                      ('38426ba3cbf5484555f2b5e9504a6b03', '6cb18e0936faa730077732a25c3dfb94'),
                      ('3877f2981e502fe1812af38d4f511fd2', '902786862cbae94e890a090e5700298b'),
                      ('43ab682adde9c14adb7c05435e5f2e0e', '9a0238499efb15551f06ad583a6fa951'),
                      ('43bd09ca68b3bcdc2b0c549fd309d1ba', '6ae42b731c00756ddd291fa615c822a1'),
                      ('54563ff51aa70ea8c6a9325c15f55399',  'b956edfd0677dd6d95de6cb29a85db9c'),
                      ('5a5c229e1340c0da7798b26edf86d180', 'dd042410dc7f02e648162d7764b50900'),
                      ('5a642ecc14e9c57a05b8e010414011f2', 'c504568822c53675a4f425c8e5800a36'),
                      ('5da97b511389a1b62ef7a55b0a19a532', '8ffde3ae7ab3726cff7ca28697687a42'),
                      ('5ef7ba98fc97917aec56ded5d5c2b099', '67e97de8ec7ddcda59a58b027263cdcc'),
                      ('68e55574e523cf1cdc17b60ce6cc2f60', '9b3267c1652691240d78b7b3d072baf3'),
                      ('72b33c9c368d86648b756143ab19baeb', '763d66b9cf01069602a968e573feb334'),
                      ('78a02b3cb6ed38b2772215c0c0a7f78e', 'c25384f6d93ca6b802925da84dfa453e'),
                      ('839087a28fa67bf97cdcaf4c8db458ef', 'a8f044478dba8040cc410e3ec7514da1'),
                      ('851c7427071afd2eaf38af0def360987', 'b49ad3aac4296376d7520445a27726de'),
                      ('871bb3cbdf48bd3bfd5a6779e752613e', '988b31dd48a1bc867dbc9e14d21b05f6'),
                      ('87c6a8f85af93b84594a36f8ffd5d6b8',  'd050e78384bd8b20e7291b3efedf6a5b'),
                      ('8f20c67f8b1230d1488138e2adbb0e64', 'b190f25b33bd52a8aae8fd81bd069888'),
                      ('9f5a457ce7e22eecd0992f4ea17b6107', 'b967656eb7e648a524ca4ffbbc172c06'),
                      ('a9513f7f0c93e179b87c01be847b3e4c', 'b86589c3e85f784a5278e377b726a4d4'),
                      ('b148cbea87c3dcc65a05b15f78910715', 'e09a818b7534422fb4c688f12566e38f'),
                      ('dbf25ce0b2a5d3cb43af95b2bd855718', 'e359704524fa26d6a3dcd8bfeeaedd2e')}    
    
    # get list of potential duplicates in validation fold
    valid_duplicates = [i for i in valid_ids if i in list(sum(duplicate_list, ()))]
    
    # get list of duplicates between train and valid
    drop_list = []
    for trn_i in train_ids:

        # find relevant pair
        val_i = None
        pair  = [pair for pair in duplicate_list if trn_i in pair]

        # find nearest neighbor
        if len(pair) > 0: 
            if pair[0][0] == trn_i:
                val_i = pair[0][1]
            else:
                val_i = pair[0][0]

        if val_i in valid_ids: 
            drop_list.append(trn_i)

    # return list of duplicates    
    return drop_list
