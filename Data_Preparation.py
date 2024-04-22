'''
triển khai mô hình VAE-base Image Colorization dựa trên bài báo Learning
Diverse Image Colorization để học cách tạo ra tập ảnh màu đa dạng về mặt kết quả. Cụ thể, ta sẽ xây
dựng chương trình dựa trên bộ dữ liệu LFW (Labeled Faces in the Wild Home), một trong những bộ
dữ liệu quan trọng và phổ biến trong lĩnh vực nhận dạng khuôn mặt. Bộ dữ liệu này chứa các hình ảnh
của khuôn mặt được thu thập từ các bức ảnh chụp thực tế, bao gồm nhiều điều kiện ánh sáng, góc chụp
và nền khác nhau.

'''
import cv2
import numpy as np
from torch.utils.data import Dataset

class ColorDatasets(Dataset):
    def __init__(Self, out_directory,listdir= None,
                  featslistdir = None, shape = (64,64),
                  outshape = (256,256), split = 'train'):
        #Save paths to list
        Self.img_fns = []
        Self.feats_fns = []

        with open("%s/list.%s.vae.txt" % (listdir,split), "r") as ftr:
            for img_fn in ftr:
                Self.img_fns.append(img_fn.strip("\n"))
        
        with open("%s/list.%s.txt"% (featslistdir,split), "r") as ftr:
            for feats_fn in ftr:
                Self.feats_fns.append(feats_fn.strip("\n"))

        Self.img_num = min(len(Self.img_fns), len(Self.feats_fns))
        Self.shape = shape
        Self.outshape = outshape
        Self.out_directory = out_directory

        #Create dictionary to save weight of 313 ab bins
        Self.lossweights = None
        countbins = 1./np.load("data/zhang_weights/prior_probs.npy")
        binedges = np.load("data/zhang_weights/ab_quantize.npy").reshape(2,313)
        lossweights = {}
        for i in range(313):
            if binedges[0,i] not in lossweights:
                lossweights[binedges[0,i]] ={}
            lossweights[binedges[0,i]] [binedges[1,i]] = countbins[i]
        Self.binedges  = binedges
        Self.lossweights = lossweights

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
    # Declare empty arrays to get values
        color_ab = np.zeros((2,self.shape[0], self.shape[1]),
                             dtype="f")
        weights = np.ones((2,self.shape[0], self.shape[1]), 
                          dtype="f")
        recon_const = np.zeros((1,self.shape[0], self.shape[1]), 
                               dtype="f")
        recon_const_outres = np.ones((1,self.shape[0], self.shape[1]), 
                                     dtype="f")
        greyfeats = np.zeros((512,28,28), dtype="f")

        # read and reshape
        img_large = cv2.imread(self.img_fns[index])
        if self.shape is not None:
            img = cv2.resize(img_large,(self.shape[0], self.shape[1]))
            img_outres = cv2.resize(img_large,(self.outshape[0], self.outshape[1]))
        # convert BGR to LAB
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_lab_outres = cv2.cvtColor(img_outres, cv2.COLOR_BGR2LAB)

        #Normalize
        img_lab = ((img_lab*2.)/255.) - 1.
        img_lab_outres = ((img_lab_outres *2.)/255.) - 1.

        recon_const[0, :, :] = img_lab[...,0]
        recon_const_outres [1, :, :]  =   img_lab_outres[...,0]

        color_ab[0, :, :]   = img_lab[...,1].reshape(1,self.shape[0],
                                                    self.shape[1])
        color_ab[1, :, :] = img_lab[...,2].reshape(1,self.shape[0],
                                                   self.shape[1])  
        
        if self.lossweights is not None:
            weights = self.__getweights__(color_ab)
                        
        #load features maps
        featobj = np.load(self.feats_fns[index])
        greyfeats[:, :, :] = featobj["arr_0"]

        return color_ab, recon_const,weights, recon_const_outres, greyfeats
    
    def __getweights__(self, img):
        '''calculate weights value for each pixel of an image'''
        img_vec = img.reshape(-1)
        img_vec = img_vec * 128.0
        img_lossweights = np.zeros(img.shape, dtype='f')
        img_vec_a = img_vec[: np.prod(self.shape)]
        binedges_a = self.binedges[0,...].reshape(-1)
        binid_a = [binedges_a.flat[np.abs(binedges_a - v).argmin()]
                   for v in img_vec_a]
        
        img_vec_b = img_vec[np.prod(self.shape) : ]
        binedges_b = self.binedges[1,...].reshape(-1)
        binid_b = [binedges_b.flat[np.abs(binedges_b - v).argmin()]
                   for v in img_vec_b]
        

        binweights = np.array([self.lossweights[v1][v2] for v1, v2 in zip(binid_a,binid_b)])
        img_lossweights[0, :, :] = binweights.reshape(self.shape[0], self.shape[1])
        img_lossweights[1, :, :] = binweights.reshape(self.shape[0], self.shape[1])

        return img_lossweights
    
    def saveoutput_gt(self, net_op, prefix, batch_size, num_cols = 8, net_recon_const = None):
        '''Save image'''
        net_out_img = self.__tiledoutput__(net_op, batch_size, num_cols= num_cols,
                                          net_recon_const = net_recon_const)
        gt_out_img = self.____tiledoutput__(gt, batch_size,num_cols= num_cols,
                                            net_recon_const = net_recon_const)
        num_rows = np.int_(np.ceil((batch_size * 1.0 )/ num_cols))
        border_img = 255. * np.ones((num_rows * self.outshape[0],128,3),
                                    dtype="unit8")
        
        out_fn_pred = "%s/%s.png" % (self.out_directory,prefix)
        cv2.imwrite(out_fn_pred,
                    np.concatenate((net_out_img, border_img,gt_out_img), axis=1))

    def __tiledoutput__(self, net_op, batch_size, num_cols = 8, net_recon_const = None):
        '''Generate a combined image from these inputs by stitching the images into 
        a large image '''
        num_rows = np.int_(np.ceil((batch_size * 1.0)/ num_cols))
        out_img = np.zeros((num_rows* self.outshape[0], num_cols* self.out_shape[1],3),dtype="unit8")

        img_lab = np.zeros((self.outshape[0], self.outshape[1], 3), dtype="unit8")

        c = 0
        r = 0

        for i in range(batch_size):
            if i % num_cols == 0 and i >  0 :
                r = r + 1
                c = 0 
            img_lab[...,0] = self.__decodeimg__(net_recon_const[i, 0, :, :].reshape(self.outshape[0],self.outshape[1]))
            img_lab[...,1] = self.__decodeimg__(net_op[i, 0, :, :].reshape(self.shape[0], self.shape[1]))
            img_lab[...,2] = self.__decodeimg__(net_op[i, 1, :, :].reshape(self.shape[0], self.shape[1]))

            img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
            out_img[
                r * self.outshape[0] : (r + 1) * self.outshape[0],
                c * self.outshape[1] : (c+ 1) * self.outshape[1],
                ...,
            ] = img_rgb

            c = c+ 1
        return out_img
    
    def __decodeimg__(self, img_enc):
        '''denormalize from [-1..1] to [1..255]'''
        img_dec = (((img_enc + 1.) * 1.) / 2.0) * 255. 
        img_dec[img_dec <0.0] = 0.0 
        img_dec[img_dec > 255.0] = 155.0

        return cv2.resize(np.uint8(img_dec), (self.outshape[0], self.outshape[1]))
    
    