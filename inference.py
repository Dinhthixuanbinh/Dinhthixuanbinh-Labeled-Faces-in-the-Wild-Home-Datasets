
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import  args, get_dirpaths
from dataloader import ColorDatasets
from VAE import VAE
from MDN import MDN
from losses import get_gmm_coeffs

def inference(vae_ckpt=None, mdn_ckpt=None):
    #Load hyperparameters
    out_dir , listdir, featslistdir  = get_dirpaths(args)
    batchsize = args['batchsize']
    hiddensize = args['hiddensize']
    nmix = args['nmix']

    #Create DataLoader
    data = ColorDatasets(
        os.path.join(out_dir, "images"),
        listdir= listdir,
        featslistdir= featslistdir,
        split= "test",
    )

    nbatches = np.int_(np.floor(data.img_num / batchsize))

    data_loader = DataLoader(
        dataset= data,
        num_workers= args['nthreads'],
        batch_size= batchsize,
        shuffle= False,
        drop_last= True,
    )

    #Load VAE model
    model_vae = VAE()
    model_vae.cuda()
    model_vae.load_state_dict(torch.load("%s/models/model_vae.pth" % (out_dir)))
    model_vae.eval()

    #Load MDN model
    model_mdn = MDN()
    model_mdn.cuda()
    model_mdn.load_state_dict(torch.load("%s/models/model_mdn.pth" % (out_dir)))
    model_mdn.eval()

    #Infer
    for batch_idx , (
        batch,
        batch_recon_const,
        batch_weights,
        batch_recon_const_outres,
        batch_feats,
    ) in tqdm (enumerate(data_loader), total= nbatches) :
        input_feats = batch_feats.cuda()

        # Get GMM parameters
        mdn_gmm_params = model_mdn(input_feats)
        gmm_mu, gmm_pi = get_gmm_coeffs(mdn_gmm_params)
        gmm_pi = gmm_pi.reshape(-1,1)
        gmm_mu = gmm_mu.reshape(-1, hiddensize)

        for j in range(batchsize):
            batch_j = np.tile(batch[j, ...].numpy(), (batchsize,1,1,1))
            batch_recon_const_j = np.tile(
                batch_recon_const[j, ...].numpy(), (batchsize,1,1,1)
            )
            batch_recon_const_outres_j = np.tile(
                batch_recon_const_outres[j, ...].numpy(), (batchsize,1,1,1)
            )

            input_color = torch.from_numpy(batch_j).cuda()
            input_greylevel= torch.from_numpy(batch_recon_const_j).cuda()

            # Get mean from GMM
            curr_mu = gmm_mu[j * nmix : (j + 1) * nmix, :]
            orderid = np.argsort(
                gmm_pi[j * nmix : (j + 1) * nmix , 0].cpu().data.numpy().reshape(-1)
            )

            # Sample from GMM
            z = curr_mu.repeat(int((batchsize * 1.0 ) /nmix), 1 )

            # Predict color 
            _, _, color_out = model_vae(input_color, input_greylevel, z)
            
            # Save image 
            data.saveoutput_gt(
                color_out.cpu().data.numpy()[orderid, ...],
                batch_j[orderid, ...],
                "divcolor_%05d_%05d" % (batch_idx, j),
                nmix,
                net_recon_const= batch_recon_const_outres_j[orderid, ...],
            )
        print("\nComplete inference. The results are saved in data/output/lfw/images.")

