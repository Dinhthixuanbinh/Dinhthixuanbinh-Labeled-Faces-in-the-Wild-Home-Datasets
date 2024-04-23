import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.hidden_size = 64

        # encoder block

        self.enc_conv1 = nn.Conv2d(2, 128, 5 , stride= 2 , padding= 2)
        self.enc_bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, 5 , stride= 2 , padding= 2)
        self.enc_bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256,512,  5 , stride= 2 , padding= 2)
        self.enc_bn3 = nn.BatchNorm2d(512)
        self.enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding= 2)
        self.enc_bn4 = nn.BatchNorm2d(1024)
        self.enc_fc1 = nn.Linear( 4 * 4 * 1024, self.hidden_size * 2)
        self.enc_dropout1 = nn.Dropout(p = 0.7)

        # Conditional encoder block 

        self.cond_enc_conv1 = nn.Conv2d (1 , 128 , 5 , stride =2 , padding =2)
        self.cond_enc_bn1 = nn.BatchNorm2d(128)
        self.cond_enc_conv2 = nn.Conv2d ( 128 , 256, 5 , stride =2 , padding =2)
        self.cond_enc_bn2 = nn.BatchNorm2d(256)
        self.cond_enc_conv3 = nn.Conv2d ( 256 , 512, 3 , stride =2 , padding =2)
        self.cond_enc_bn3 = nn.BatchNorm2d(512)
        self.cond_enc_conv4 = nn.Conv2d ( 512 , 1024, 3 , stride =2 , padding =2)
        self.cond_enc_bn4 = nn.BatchNorm2d(1024)

        # Decoder block

        
