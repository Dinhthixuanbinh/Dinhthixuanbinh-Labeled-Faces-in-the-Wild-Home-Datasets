

import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.optim as optim

from hyperparamaters import  args, get_dirpaths
from Data_Preparation import ColorDatasets
from VAE_model import vae_loss, VAE
def test_vae(model):
    model.eval()

    #load hyperparameters
    out_dir , listdir, featslistdir = get_dirpaths(args)
    batchsize = args['batchsize']
    hiddensize = args['hiddensize']
    nmix = args['nmix']

    #create dataloader
    data = ColorDatasets(

    os.path.join(out_dir, "image"),
    listdir= listdir,
    featslistdir= featslistdir,
    split= "test",
   )
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(
        dataset= data,
        num_workers= args["nthreats"],
        batch_size= batchsize,
        shuffle= False,
        drop_last= True,
    )

    # Eval
    test_loss = 0.0
    for batch_idx ,(
        batch , 
        batch_recon_const, 
        batch_weights, 
        batch_recon_const_outres,
        _,) in tqdm (enumerate(data_loader), total= nbatches): 
        input_color = batch.cuda()
        lossweights = batch_weights.cuda()
        lossweights = lossweights.reshape(batchsize, -1)
        input_greylevel = batch_recon_const.cuda()
        z = torch.randn(batchsize, hiddensize)

        mu, logvar, color_out, = model(input_color, input_greylevel, z)
        _, _, recon_loss_12  = vae_loss (mu, logvar, color_out, input_color, lossweights, batchsize)
        test_loss = test_loss + recon_loss_12.item()
    test_loss = (test_loss *1.0) / nbatches
    model.train()

    return test_loss
    
def train_vae():
    # Load hyperparameters
    out_dir, listdir, featslistdir = get_dirpaths(args)
    batchsize = args['batchsize']
    hiddensize = args['hiddensize']
    nmix = args['nmix']
    nepochs = args['epochs']

    #create DataLoader

    data = ColorDatasets(
        os.path.join(out_dir, "image"),
        listdir= listdir,
        featslistdir= featslistdir,
        split= "train",
    )
    nbatches = np.int_(np.floor(data.img_num / batchsize))
    data_loader = DataLoader(
        dataset= data,
        num_workers= args['nthreads'],
        batch_size=batchsize,
        shuffle= True,
        drop_last= True,
    )

    # Initialize VAE model
    model = VAE()
    model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr = 5e-5)

    #Train 
    itr_idx = 0
    for epochs in range(nepochs):
        train_loss = 0.0

        for batch_idx,(
            batch,
            batch_recon_const,
            batch_weghts,
            batch_recon_const_outres,
            _,
        ) in tqdm (enumerate(data_loader), total= nbatches):
            
            input_color = batch.cuda()
            lossweights = batch_weghts.cuda()
            lossweights = lossweights.reshape(batchsize, -1 )
            input_greylevel = batch_recon_const.cuda()
            z = torch.rand(batchsize, hiddensize)

            optimizer.zero_grad()
            mu, logvar, color_out, = model(input_color, input_greylevel, z) 
            kl_loss, recon_loss, recon_loss_12 = vae_loss(mu, logvar, color_out, input_color, lossweights, batchsize)
            loss = kl_loss.mul(1*e - 2) + recon_loss
            recon_loss_12.detach()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + recon_loss_12.item()

            if batch_idx % args['logstep'] == 0 :
                data.saveoutput_gt(
                    color_out.cpu().data.numpy(),
                    batch.numpy(),
                    "train_%05d_%05d" % (epochs, batch_idx),
                    batchsize,
                    net_recon_const= batch_recon_const_outres.numpy(),
                )
        train_loss = (train_loss * 1.0) /(nbatches)
        print("VAE Train Losss, epoch %d has loss %f" % (epochs, train_loss))
        test_loss =  test_vae(model)
        print("VAE Test Losss, epoch %d has loss %f" % (epochs, test_loss))

        #Save VAE model 
        torch.save(model.state_dict(), "%s/models/model_vae.pth" % (out_dir))
    print("Complete VAE training")
train_vae()