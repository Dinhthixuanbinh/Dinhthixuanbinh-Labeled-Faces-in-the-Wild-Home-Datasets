

from hyperparamaters import  args, get_dirpaths
from Data_Preparation import ColorDatasets
def test_vae(model):
    model.eval()

    #load hyperparameters
    out_dir , listdir, featslistdir = get_dirpaths(args)
    batchsize = args['batchsize']
    hiddensize = args['hiddensize']
    nmix = args['nmix']

    #create dataloader
    data = ColorDatasets(

        
    )