import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch



def get_augs(CFG, image_size = None, p_aug = None):
    
    '''
    Get train and test augmentations
    '''

    # update epoch-based parameters
    if image_size is None:
        image_size = CFG['image_size']
    if p_aug is None:
        p_aug = CFG['p_aug']
        
    # random crop
    if CFG['crop_scale'] == (1, 1):
        p_crop = 0
    else:
        p_crop = 1
        
    # blur
    if CFG['blur_limit'] == 0:
        p_blur = 0
    else:
        p_blur = p_aug
        
    # distortion
    if CFG['dist_limit'] == 0:
        p_dist = 0
    else:
        p_dist = p_aug
        
    # normalization
    if not CFG['normalize']:
        CFG['pixel_mean'] = (0, 0, 0)
        CFG['pixel_std']  = (1, 1, 1)
    else:
        CFG['pixel_mean'] = (0.485, 0.456, 0.406)
        CFG['pixel_std']  = (0.229, 0.224, 0.225)
        
    # train augmentations
    train_augs = A.Compose([A.SmallestMaxSize(max_size = 512),
                            #A.Resize(p       = 1 - p_crop,
                            #         height  = image_size, 
                            #         width   = image_size),
                            #A.RandomResizedCrop(p      = p_crop,
                            #                    height = image_size, 
                            #                    width  = image_size,
                            #                    scale  = CFG['crop_scale']),
                            A.Transpose(p      = CFG['p_transpose']),
                            A.HorizontalFlip(p = CFG['p_hflip']),
                            A.VerticalFlip(p   = CFG['p_vflip']),
                            A.ShiftScaleRotate(p            = p_aug,
                                               shift_limit  = CFG['ssr'][0],
                                               scale_limit  = CFG['ssr'][1],
                                               rotate_limit = CFG['ssr'][2]),
                            A.HueSaturationValue(p               = p_aug,
                                                 hue_shift_limit = CFG['huesat'][0],
                                                 sat_shift_limit = CFG['huesat'][1],
                                                 val_shift_limit = CFG['huesat'][2]),
                            A.RandomBrightnessContrast(p                = p_aug,
                                                       brightness_limit = CFG['bricon'][0],
                                                       contrast_limit   = CFG['bricon'][1]),
                            A.CLAHE(p              = p_aug,
                                    clip_limit     = CFG['clahe'][0],
                                    tile_grid_size = (CFG['clahe'][1], CFG['clahe'][1])),
                            A.OneOf([A.MotionBlur(blur_limit   = CFG['blur_limit']),
                                     A.GaussianBlur(blur_limit = CFG['blur_limit'])], 
                                     p                         = p_blur),
                            A.OneOf([A.OpticalDistortion(distort_limit = CFG['dist_limit']),
                                     A.GridDistortion(distort_limit    = CFG['dist_limit'])], 
                                     p = p_dist),
                            A.Resize(p       = 1 - p_crop,
                                     height  = image_size, 
                                     width   = image_size),
                            A.RandomResizedCrop(p      = p_crop,
                                                height = image_size, 
                                                width  = image_size,
                                                scale  = CFG['crop_scale']),
                            A.Cutout(p          = p_aug, 
                                     num_holes  = CFG['cutout'][0], 
                                     max_h_size = np.int(CFG['cutout'][1] * image_size), 
                                     max_w_size = np.int(CFG['cutout'][1] * image_size)),
                            A.Normalize(mean = CFG['pixel_mean'],
                                        std  = CFG['pixel_std']),
                            ToTensorV2()
                           ])

    # valid augmentations
    valid_augs = A.Compose([A.SmallestMaxSize(max_size = image_size),
                            A.CenterCrop(height = image_size, 
                                         width  = image_size),
                            #A.Resize(height  = image_size, 
                            #         width   = image_size),
                            A.Normalize(mean = CFG['pixel_mean'],
                                        std  = CFG['pixel_std']),
                            ToTensorV2()
                           ])
    
    # output
    return train_augs, valid_augs



####### TTA FLIPS
 
def get_tta_flips(img, i):
    
    '''
    Get TTA flips
    Based on https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
    '''

    if i >= 4:
        img = img.transpose(2, 3)
    if i % 4 == 0:
        return img
    elif i % 4 == 1:
        return img.flip(3)
    elif i % 4 == 2:
        return img.flip(2)
    elif i % 4 == 3:
        return img.flip(3).flip(2)
    
    
    
####### CUTMIX

def rand_bbox(size, lam):
    
    '''
    Random image box for cutmix
    '''

    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w   = np.int(W * cut_rat)
    cut_h   = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_fn(data, target, alpha):
    
    '''
    Cutmix augmentation
    '''

    indices       = torch.randperm(data.size(0))
    shuffled_data   = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets