# Declare hyperparamaters

args = {
    "gpu": 1,
    "epochs" : 2,
    "batchsize": 32,
    "hiddensize" : 64,
    "nthreads" : 2 ,
    "epochs_mdn": 2,
    "nmix" : 8,
    "logstep":100,
    "dataset_key": "lfw"
    } 
def get_dirpaths(args):
    if args["dataset_key"] == "lfw":
        out_dir = "data/output/lfw"
        listdir = "data/imglist/lfw"
        featslistdir = "data/featslist/lfw"
    else:
        raise NameError("[ERROR] Incorrect key : %s" % (args.dataset_key))
    return out_dir, listdir,featslistdir
