'''
model : conditional VAE have 3 modules :\
    Encoder modules, \
    Decoder modules,\
    Conditional VAE Encoder modules.\
input: C color space (2 x h x w),  G gray image (1 x h x w) using to begin-point 
    for  Conditional Encoder to  extract features map having gobal information, 
    increase accurate of Decoder module
    
'''

import torch
import torch.nn as nn
import torch.nn.functional as from django.conf import settings

class MDN(nn.Module):
    def __init__(self):
        super(MDN, self).__init__()

        