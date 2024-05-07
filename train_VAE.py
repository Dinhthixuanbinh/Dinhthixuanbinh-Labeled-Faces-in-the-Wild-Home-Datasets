

import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.optim as optim

from config import  args, get_dirpaths
from dataloader import ColorDatasets
from VAE import VAE
from losses import vae_loss

def test_vae(model):
    model.eval()

    # Load hyperparameters
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args["batchsize"]
    hiddensize = args["hiddensize"]
    nmix = args["nmix"]

    # Create DataLoader
    data = ColorDatasets(os.path.join(out_dir, "images"), listdir, featslistdir, split="test")
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(dataset=data, num_workers=args["nthreads"], batch_size=batchsize, shuffle=False, drop_last=True)

    # Eval
    test_loss = 0.0
    for batch_idx, (batch, batch_recon_const, batch_weights, batch_recon_const_outres, _) in tqdm(enumerate(data_loader), total=nbatches):
        input_color = batch.cuda()
        lossweights = batch_weights.cuda()
        lossweights = lossweights.reshape(batchsize, -1)
        input_greylevel = batch_recon_const.cuda()
        z = torch.randn(batchsize, hiddensize)

        mu, logvar, color_out = model(input_color, input_greylevel, z)
        _, _, recon_loss_l2 = vae_loss(mu, logvar, color_out, input_color, lossweights, batchsize)
        test_loss = test_loss + recon_loss_l2.item()

    test_loss = (test_loss * 1.0) / nbatches
    model.train()
    return test_loss


def train_vae():
    # Load hyperparameters
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args["batchsize"]
    hiddensize = args["hiddensize"]
    nmix = args["nmix"]
    nepochs = args["epochs"]

    # Create DataLoader
    data = ColorDatasets(os.path.join(out_dir, "images"), listdir, featslistdir, split="train")
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(dataset=data, num_workers=args["nthreads"], batch_size=batchsize, shuffle=True, drop_last=True)

    # Initialize VAE model
    model = VAE()
    model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # Train
    for epochs in range(nepochs):
        train_loss = 0.0

        for batch_idx, (batch, batch_recon_const, batch_weights, batch_recon_const_outres, _) in tqdm(enumerate(data_loader), total=nbatches):
            input_color = batch.cuda()
            lossweights = batch_weights.cuda()
            lossweights = lossweights.reshape(batchsize, -1)
            input_greylevel = batch_recon_const.cuda()
            z = torch.randn(batchsize, hiddensize)

            optimizer.zero_grad()
            mu, logvar, color_out = model(input_color, input_greylevel, z)
            kl_loss, recon_loss, recon_loss_l2 = vae_loss(mu, logvar, color_out, input_color, lossweights, batchsize)
            loss = kl_loss.mul(1e-2) + recon_loss
            recon_loss_l2.detach()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + recon_loss_l2.item()

            if batch_idx % args["logstep"] == 0:
                data.saveoutput_gt(
                    color_out.cpu().data.numpy(),
                    batch.numpy(),
                    "train_%05d_%05d" % (epochs, batch_idx),
                    batchsize,
                    net_recon_const=batch_recon_const_outres.numpy()
                )

        train_loss = (train_loss * 1.0) / (nbatches)
        test_loss = test_vae(model)
        print(f"End of epoch {epochs:3d} | Train Loss {train_loss:8.3f} | Test Loss {test_loss:8.3f} ")

        # Save VAE model
        torch.save(model.state_dict(), "%s/models/model_vae.pth" % (out_dir))

    print("Complete VAE training")
train_vae()