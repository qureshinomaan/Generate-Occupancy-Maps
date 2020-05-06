#======================================================================================#
# Importing all the important libraries.
#======================================================================================#
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#======================================================================================#

#======================================================================================#
# ResNet Model #
# ResNet 50 #
# https://www.google.com/search?q=convolution+dimension+formula&safe=active&client=firefox-b-d&sxsrf=ALeKk038gpXP7ihgPbHHEnhnYPV-FgjoIA:1588603828017&source=lnms&tbm=isch&sa=X&ved=2ahUKEwir8O22uprpAhUyIbcAHbOODSgQ_AUoAXoECA4QAw&biw=1280&bih=703#
#======================================================================================#
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=64, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2)
        
        # Stage 1 #
        self.convStage111 = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=1, stride=1)
        self.convStage112 = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride=1)
        self.convStage113 = nn.Conv2d(in_channels = 64, out_channels=256, kernel_size=1, stride=1)
        
        self.convStage121 = nn.Conv2d(in_channels = 256, out_channels=64, kernel_size=1, stride=1)
        self.convStage122 = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride=1)
        self.convStage123 = nn.Conv2d(in_channels = 64, out_channels=256, kernel_size=1, stride=1)
        
        self.convStage131 = nn.Conv2d(in_channels = 256, out_channels=64, kernel_size=1, stride=1)
        self.convStage132 = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride=1)
        self.convStage133 = nn.Conv2d(in_channels = 64, out_channels=256, kernel_size=1, stride=1)
        
        self.convskip11 = nn.Conv2d(in_channels = 64, out_channels=256, kernel_size=3, stride=1)
        self.convskip12 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, stride=1)
        self.convskip13 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, stride=1)
        
        #Stage 2#
        self.convStage211 = nn.Conv2d(in_channels = 256, out_channels=128, kernel_size=1, stride=2)
        self.convStage212 = nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=3, stride=1)
        self.convStage213 = nn.Conv2d(in_channels = 128, out_channels=512, kernel_size=1, stride=1)
        
        self.convStage221 = nn.Conv2d(in_channels = 512, out_channels=128, kernel_size=1, stride=1)
        self.convStage222 = nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=3, stride=1)
        self.convStage223 = nn.Conv2d(in_channels = 128, out_channels=512, kernel_size=1, stride=1)
        
        self.convStage231 = nn.Conv2d(in_channels = 512, out_channels=128, kernel_size=1, stride=1)
        self.convStage232 = nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=3, stride=1)
        self.convStage233 = nn.Conv2d(in_channels = 128, out_channels=512, kernel_size=1, stride=1)
        
        self.convStage241 = nn.Conv2d(in_channels = 512, out_channels=128, kernel_size=1, stride=1)
        self.convStage242 = nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=3, stride=1)
        self.convStage243 = nn.Conv2d(in_channels = 128, out_channels=512, kernel_size=1, stride=1)
        
        self.convskip21 = nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=5, stride=2)
        self.convskip22 = nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=3, stride=1)
        self.convskip23 = nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=3, stride=1)
        self.convskip24 = nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=3, stride=1)
        
        #stage 3#
        self.convStage311 = nn.Conv2d(in_channels = 512, out_channels=256, kernel_size=1, stride=2)
        self.convStage312 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, stride=1)
        self.convStage313 = nn.Conv2d(in_channels = 256, out_channels=1024, kernel_size=1, stride=1)
        
        self.convStage321 = nn.Conv2d(in_channels = 1024, out_channels=256, kernel_size=1, stride=1)
        self.convStage322 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, stride=1)
        self.convStage323 = nn.Conv2d(in_channels = 256, out_channels=1024, kernel_size=1, stride=1)
        
        self.convStage331 = nn.Conv2d(in_channels = 1024, out_channels=256, kernel_size=1, stride=1)
        self.convStage332 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, stride=1)
        self.convStage333 = nn.Conv2d(in_channels = 256, out_channels=1024, kernel_size=1, stride=1)
        
        self.convStage341 = nn.Conv2d(in_channels = 1024, out_channels=256, kernel_size=1, stride=1)
        self.convStage342 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, stride=1)
        self.convStage343 = nn.Conv2d(in_channels = 256, out_channels=1024, kernel_size=1, stride=1)
        
        self.convStage351 = nn.Conv2d(in_channels = 1024, out_channels=256, kernel_size=1, stride=1)
        self.convStage352 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, stride=1)
        self.convStage353 = nn.Conv2d(in_channels = 256, out_channels=1024, kernel_size=1, stride=1)
        
        self.convStage361 = nn.Conv2d(in_channels = 1024, out_channels=256, kernel_size=1, stride=1)
        self.convStage362 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, stride=1)
        self.convStage363 = nn.Conv2d(in_channels = 256, out_channels=1024, kernel_size=1, stride=1)
        
        self.convskip31 = nn.Conv2d(in_channels = 512, out_channels=1024, kernel_size=5, stride=2)
        self.convskip32 = nn.Conv2d(in_channels = 1024, out_channels=1024, kernel_size=3, stride=1)
        self.convskip33 = nn.Conv2d(in_channels = 1024, out_channels=1024, kernel_size=3, stride=1)
        self.convskip34 = nn.Conv2d(in_channels = 1024, out_channels=1024, kernel_size=3, stride=1)
        self.convskip35 = nn.Conv2d(in_channels = 1024, out_channels=1024, kernel_size=3, stride=1)
        self.convskip36 = nn.Conv2d(in_channels = 1024, out_channels=1024, kernel_size=3, stride=1)
        
        #Stage 4#
        self.convStage411 = nn.Conv2d(in_channels = 1024, out_channels=512, kernel_size=1, stride=2)
        self.convStage412 = nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=3, stride=1)
        self.convStage413 = nn.Conv2d(in_channels = 512, out_channels=2048, kernel_size=1, stride=1)
        
        self.convStage421 = nn.Conv2d(in_channels = 2048, out_channels=512, kernel_size=1, stride=1)
        self.convStage422 = nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=3, stride=1)
        self.convStage423 = nn.Conv2d(in_channels = 512, out_channels=2048, kernel_size=1, stride=1)
        
        self.convStage421 = nn.Conv2d(in_channels = 2048, out_channels=512, kernel_size=1, stride=1)
        self.convStage422 = nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=3, stride=1)
        self.convStage423 = nn.Conv2d(in_channels = 512, out_channels=2048, kernel_size=1, stride=1)
        
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride = 2)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))                     # First  Layer
        x = self.maxpool(x)                           # First  Layer
        
        prev = x
        prev = self.convskip11(prev)
        x = F.relu(self.convStage111(x))
        x = F.relu(self.convStage112(x))
        x = F.relu(self.convStage113(x) + prev)
        prev = self.convskip12(prev)
        
        x = F.relu(self.convStage121(x))
        x = F.relu(self.convStage122(x))
        x = F.relu(self.convStage123(x) + prev)
        prev = self.convskip13(x)
        
        x = F.relu(self.convStage131(x))
        x = F.relu(self.convStage132(x))
        x = F.relu(self.convStage133(x) + prev)
        prev = self.convskip21(x)
        
        x = F.relu(self.convStage211(x))
        x = F.relu(self.convStage212(x))
        x = F.relu(self.convStage213(x) + prev)
        prev = self.convskip22(x)
        
        x = F.relu(self.convStage221(x))
        x = F.relu(self.convStage222(x))
        x = F.relu(self.convStage223(x) + prev)
        prev = self.convskip23(x)
        
        x = F.relu(self.convStage231(x))
        x = F.relu(self.convStage232(x))
        x = F.relu(self.convStage233(x) + prev)
        prev = self.convskip24(x)
        
        x = F.relu(self.convStage241(x))
        x = F.relu(self.convStage242(x))
        x = F.relu(self.convStage243(x) + prev)
        prev = self.convskip31(x)
        
        x = F.relu(self.convStage311(x))
        x = F.relu(self.convStage312(x))
        x = F.relu(self.convStage313(x) + prev)
        prev = self.convskip32(x)
        
        x = F.relu(self.convStage321(x))
        x = F.relu(self.convStage322(x))
        x = F.relu(self.convStage323(x) + prev)
        prev = self.convskip33(x)
        
        x = F.relu(self.convStage331(x))
        x = F.relu(self.convStage332(x))
        x = F.relu(self.convStage333(x) + prev)
        print(x.shape)
        prev = self.convskip34(x)
        
        
        
        return x
#======================================================================================#




#======================================================================================#
# Trying Forward Propagation #
#======================================================================================#
net = Net()
net.double()
image = np.zeros(shape=(1, 3, 224, 224))
image = torch.from_numpy(image).double()
output = net(image)
#======================================================================================#
