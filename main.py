import os
from config import args
from inference import inference
from train_VAE import train_vae
from train_MDN import train_mdn

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_vae()
    train_mdn()
    inference()