from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import *
import cv2

model = stackhourglass(192)

model = nn.DataParallel(model)
model.cuda()

#loadmodel = './trained/pretrained_model_KITTI2015.tar'
loadmodel = './models/PSMNet/trained/pretrained_model_KITTI2015.tar'
print('load PSMNet')
state_dict = torch.load(loadmodel)
model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if torch.cuda.is_available():
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            disp = model(imgL,imgR)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp


def generate_disparity(leftimg, rightimg, isgray):
       processed = preprocess.get_transform(augment=False)
       if isgray:
           imgL_o = cv2.cvtColor(cv2.imread(leftimg,0), cv2.COLOR_GRAY2RGB)
           imgR_o = cv2.cvtColor(cv2.imread(rightimg,0), cv2.COLOR_GRAY2RGB)
       else:
           imgL_o = (skimage.io.imread(leftimg).astype('float32'))
           imgR_o = (skimage.io.imread(rightimg).astype('float32'))
       
       imgL = processed(imgL_o).numpy()
       imgR = processed(imgR_o).numpy()
       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

       # pad to width and hight to 16 times
       if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16       
            top_pad = (times+1)*16 -imgL.shape[2]
       else:
            top_pad = 0
       if imgL.shape[3] % 16 != 0:
            times = imgL.shape[3]//16                       
            left_pad = (times+1)*16-imgL.shape[3]
       else:
            left_pad = 0     
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       start_time = time.time()
       pred_disp = test(imgL,imgR)
       print('time = %.2f' %(time.time() - start_time))
       if top_pad !=0 or left_pad != 0:
            img = pred_disp[top_pad:,:-left_pad]
       else:
            img = pred_disp
       img = (img*256).astype('uint16')
       skimage.io.imsave('./outputs/disparity/disparity.png',img)
       
       #img = np.concatenate((imgL_o, imgR_o),axis=1)
       #img = cv2.line(img, (0, 240), (1504, 240), (0, 0, 255), 2)
       #img = cv2.line(img, (0, 210), (1504, 210), (0, 0, 255), 2)
       #img = cv2.line(img, (0, 270), (1504, 270), (0, 0, 255), 2)
       #skimage.io.imsave('test.png',img)





