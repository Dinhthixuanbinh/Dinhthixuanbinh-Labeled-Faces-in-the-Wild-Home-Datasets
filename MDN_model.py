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
import torch.nn.functional as F

from hyperparamaters import args 

class MDN(nn.Module):
    def __init__(self):
        super(MDN, self).__init__()

        self.feats_nch = 512
        self.hidden_size = 64
        self.nmix = 8
        self.nout = (self.hidden_size + 1) * self.nmix

        self.model = nn.Sequential(
            nn.Conv2d(self.feats_nch,384,5, stride=1, padding=2),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384,320,5,stride=1,padding=2),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(320,288,5,stride=1,padding=2),
            nn.BatchNorm2d(288),
            nn.ReLU(),
            nn.Conv2d(288,256,5,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,128,5,stride=1,padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,96,5,stride=1,padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96,64,5,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.7)
        )

        self.fc = nn.Linear(4 * 4 * 64, self.nout)

    def forward(self, feats): 
        x = self.model(feats)
        x = x.view(-1,4, * 4 * 64)
        x = F.relu(x)
        x = F.dropout(x, p= 0.7, training= self.training)

        return x
    
def get_gmm_coeffs(gmm_params):
    '''
    Return the distribution coefficients of the GMM
    '''

    gmm_mu = gmm_params[..., : args['hiddensize'] * args['nmix']]
    gmm_mu.contiguous()
    gmm_pi_activ = gmm_params[..., args['hiddensize']* args['nmix'] :]
    gmm_pi_activ.contiguous()
    gmm_pi = F.softmax(gmm_pi_activ, dim= 1)

    return gmm_mu,gmm_pi

def mdn_loss(gmm_params, mu,  stddev, batchsize):
    '''
    Calculates the loss by comparing two distribution
    the predicted distribution of the MDN ( given by gmm_mu and gmm_pi ) with
    the target distribution created by the Encoder block ( given by mu and
stddev ).
    
    '''
    gmm_mu , gmm_pi = get_gmm_coeffs(gmm_params)
    eps = torch.randn(stddev.sive()).normal_().cuda()
    z = torch.add(mu, torch.mul(eps, stddev))
    z_flat = z.repeat(1, args['nmix'])
    z_flat = z_flat.reshape(batchsize * args['nmix'], args['hiddensize'])
    gmm_mu_flat = gmm_mu.reshape(atchsize * args['nmix'], args['hiddensize'])
    dist_all = torch.sqrt(torch.sum(torch.add(z_flat, gmm_mu_flat.mul(-1)).pow(2).mul(50), 1))
    dist_all = dist_all.reshape(batchsize, args['nmix'])
    dist_min, selectids = torch.min(dist_all,1)
    gmm_pi_min = torch.gather(gmm_pi,1, selectids.reshape(-1,1))
    gmm_loss = torch.mean(torch.add(-1* torch.log(gmm_pi_min + 1e-30),dist_min))
    gmm_loss_12 = torch.mean(dist_min)

    return gmm_loss, gmm_loss_12