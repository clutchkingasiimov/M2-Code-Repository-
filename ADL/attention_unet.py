import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class Conv_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Conv_block,self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        x = self.conv(x)
        return x
    
class Up_Conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Up_Conv,self).__init__()
        self.up = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
        
    def forward(self,x):
        x = self.up(x)
        return x
    
#Attention block to capture spatial information   
class Attention_Block(nn.Module):
    def __init__(self,F_gain,F_loss,F_int):
        super(Attention_Block,self).__init__()
        self.W_g = nn.Sequential(
        nn.Conv2d(F_gain,F_int,kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(F_int))
        
        self.W_x = nn.Sequential(
        nn.Conv2d(F_loss,F_int,kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(F_int))
        
        self.psi = nn.Sequential(
        nn.Conv2d(F_int,1,kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(1),
        nn.Softmax(dim=1))
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi =  self.psi(psi)
        
        return x*psi
    
    
class Attention_Unet(nn.Module):
    def __init__(self,img_channels=3,output_channels=25):
        super(Attention_Unet,self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv1 = Conv_block(img_channels,64)
        self.conv2 = Conv_block(64,128)
        self.conv3 = Conv_block(128,256)
        self.conv4 = Conv_block(256,512)
#         self.conv5 = Conv_block(512,1024)
        
#         self.up5 = Up_Conv(1024,512)
#         self.att5 = Attention_Block(512,512,256)
#         self.up_conv5 = Conv_block(1024,512)
        
        self.up4 = Up_Conv(512,256)
        self.att4 = Attention_Block(256,256,128)
        self.up_conv4 = Conv_block(512,256)
        
        self.up3 = Up_Conv(256,128)
        self.att3 = Attention_Block(128,128,64)
        self.up_conv3 = Conv_block(256,128)
        
        self.up2 = Up_Conv(128,64)
        self.att2 = Attention_Block(64,64,32)
        self.up_conv2 = Conv_block(128,64)
        
        self.convlast = nn.Conv2d(64,output_channels,1,1,0)
        self.output = nn.Softmax(dim=1)
        
        
    def forward(self,x):
        #Encoding 
        x1 = self.conv1(x)
        
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

#         x5 = self.maxpool(x4)
#         x5 = self.conv5(x5)

        # decoding + concat path
#         d5 = self.up5(x5)
#         x4 = self.att5(g=d5,x=x4)
#         d5 = torch.cat((x4,d5),dim=1)        
#         d5 = self.up_conv5(d5)
        
        d4 = self.up4(x4)
        x3 = self.att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.convlast(d2)
        output = self.output(d1)
        

        return d1