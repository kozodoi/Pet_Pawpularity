import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augs(CFG):

    '''
    Get augmentations
    '''
    
    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'

    # normalization
    if not CFG['normalize']:
        CFG['pixel_mean'] = (0, 0, 0)
        CFG['pixel_std']  = (1, 1, 1)
    else:
        CFG['pixel_mean'] = (0.485, 0.456, 0.406)
        CFG['pixel_std']  = (0.229, 0.224, 0.225)

    # valid augmentations
    valid_augs = A.Compose([A.Resize(height  = CFG['image_size'],
                                     width   = CFG['image_size']),
                            A.Normalize(mean = CFG['pixel_mean'],
                                        std  = CFG['pixel_std']),
                            ToTensorV2()
                           ])

    # output
    return valid_augs