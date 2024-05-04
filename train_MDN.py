import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.optim as optim

from hyperparamaters import  args, get_dirpaths
from Data_Preparation import ColorDatasets
from MDN_model import mdn_loss, MDN, get_gmm_coeffs
from VAE_model import VAE
def test_mdn(moddel_vae,model_mdn):
    # Load hyperparameters
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args['batchsize']
    hiddensize = args['hiddensize']
    nmix = args['nmix']

    #create DataLoader 
    data = ColorDatasets(
        os.path.join(out_dir, "images"),listdir,featslistdir,
    split= "test",
    )
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(
        dataset= data,
        num_workers= args["nthreats"],
        batch_size= batchsize,
        shuffle= True,
        drop_last= True,
    )
    optimizer = optim.Adam(model_mdn.parameters(), lr = 1e-3)

    #Eval
    moddel_vae.eval()
    model_mdn.eval()
    itr_idx = 0
    test_loss = 0.0

    for batch_idx ,(
        batch , 
        batch_recon_const, 
        batch_weights, 
        _,
        batch_feats,) in tqdm (enumerate(data_loader), total= nbatches): 
        input_color = batch.cuda()
        input_greylevel = batch_recon_const.cuda()
        input_feats = batch_feats.cuda()
    
        z = torch.randn(batchsize, hiddensize)
        optimizer.zero_grad()

        # Get the parameters of the posterior distribution

        mu, logvar, _ = moddel_vae(input_color, input_greylevel,z)

        #  Get the GMM vector
        mdn_gmm_params = model_mdn(input_feats)

        # Compare 2 distributions

        loss, _ = mdn_loss(mdn_gmm_params, mu, torch.sqrt(torch.exp(logvar)), batchsize )

        test_loss = test_loss + loss.item()

    test_loss = (test_loss * 1.0) / (nbatches)
    moddel_vae.train()
    return test_loss

def train_mdn():
    # Load hyperparameters
    out_dir , listdir, featslistdir = get_dirpaths(args)
    batchsize = args['batchsize']
    hiddensize = args['hiddensize']
    nmix = args['nmix']
    nepochs = args['epochs_mdn']

    # Create DataLoader
    data = ColorDatasets(
        os.path.join(out_dir, "images"),
        listdir,
        featslistdir,
        split= "train",
    )
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(
        dataset= data,
        num_workers= args['nthreads'],
        batch_size= batchsize,
        shuffle= True,
        drop_last= True,
    )

    # Initialize VAE model
    model_vae = VAE()
    model_vae.cuda()
    model_vae.load_state_dict(torch.load("%s/models/model_vae.pth" % (out_dir)))
    # Initialize MDN model
    model_mdn = MDN()
    model_mdn.cuda()
    model_mdn.train()

    optimizer = optim.Adam(model_mdn.parameters(), lr= 1e-3)
    # Train
    itr_idx = 0
    for epochs_mdn in range(nepochs):
        train_loss = 0.0
        for batch_idx ,(
            batch,
            batch_recon_const,
            batch_weights,
            _,
            batch_feats,
        ) in tqdm (enumerate(data_loader), total= nbatches):
            input_color = batch.cuda()
            input_greylevel = batch_recon_const.cuda()
            input_feats = batch_feats.cuda()
            z = torch.randn(batchsize, hiddensize)
            optimizer.zero_grad()

            # Get the parameters of the posterior distribution
            mu, logvar, _ = model_vae(input_color, input_greylevel, z)

            # Get the GMM vector 
            mdn_gmm_params = model_mdn(input_feats)

            # Compare 2 distributions
            loss, loss_12 = mdn_loss(
                mdn_gmm_params, mu, torch.sqrt(torch.exp(logvar)), batchsize
            )

            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()

        train_loss = (train_loss * 1.0) / (nbatches)
        test_loss = test_mdn(model_vae, model_mdn)
        print(
            f"End of epoch { epochs_mdn:3d} | Train Loss { train_loss:8.3f} | Test
Loss { test_loss:8.3f}"
        )

        # Save MDN model 
        torch.save(model_mdn.state_dict(), "%s/models_mdn/model_mdn.pth" % (out_dir))

    print("Complete MDN training")
train_mdn()

    
