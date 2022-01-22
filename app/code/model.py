import timm
import torch
import torch.nn as nn


def get_model(CFG, pretrained = None):
    
    # model class
    class Model(nn.Module):

        def __init__(self, CFG, pretrained):
            super().__init__()
            self.backbone = timm.create_model(model_name  = CFG['backbone'], 
                                              pretrained  = True if pretrained == 'imagenet' else False,
                                              in_chans    = 3,
                                              num_classes = 0)
            if CFG['loss_fn'] == 'MSE':
                self.fc = nn.Sequential(nn.LazyLinear(CFG['target_size']),
                                        nn.Sigmoid())
            else:
                self.fc = nn.Sequential(nn.LazyLinear(CFG['target_size']))
                
        def forward(self, image, features = None):
            x = self.backbone(image)
            if features is not None:
                x = torch.cat([x, features], dim = 1)
            x = self.fc(x)
            return x
        
    # pretrained weights
    if pretrained is None:
        pretrained = CFG['pretrained']
                      
    # get model
    model = Model(CFG, pretrained)
            
    # load pre-trained weights
    if pretrained:
        if pretrained != 'imagenet':
            model.load_state_dict(torch.load(pretrained, map_location = torch.device('cpu')))

    return model