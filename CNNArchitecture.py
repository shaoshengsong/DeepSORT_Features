#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
def setup_seed(seed):
     torch.manual_seed(seed)

     



class Residual4(nn.Module): # Residual(32,32)  
    def __init__(self, input, output):
        super(Residual4,self).__init__()
        self.conv1 = nn.Conv2d(input, output, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(output,output,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output)
        
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)     
        y = self.conv2(y)
        y = self.bn2(y)
        return F.relu(x.add(y),True)

class Residual5(nn.Module):#Residual(32,32)
    def __init__(self, input, output):
        super(Residual5,self).__init__()
        self.conv1 = nn.Conv2d(input, output, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(output,output,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output)
        
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)        
        y = self.conv2(y)
        y = self.bn2(y)
        return F.relu(x.add(y),True)
    
class Residual6(nn.Module):#Residual(32,64)
    def __init__(self, input, output):
        super(Residual6,self).__init__()
        self.conv1 = nn.Conv2d(input, output, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(output,output,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output)

        self.downsample = nn.Sequential(
                nn.Conv2d(input, output, 1, stride=2, bias=False),
                nn.BatchNorm2d(output),)
            
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        x = self.downsample(x)
        return F.relu(x.add(y),True)#    


class Residual7(nn.Module):#Residual(32,64)
    def __init__(self, input, output):
        super(Residual7,self).__init__()
        self.conv1 = nn.Conv2d(input, output, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(output,output,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output)

            
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        return F.relu(x.add(y),True)#  


class Residual8(nn.Module):#(64,128)
    def __init__(self, input, output):
        super(Residual8,self).__init__()
        self.conv1 = nn.Conv2d(input, output, 3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(output,output,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output)

        self.downsample = nn.Sequential(
                nn.Conv2d(input, output, 1, stride=2, bias=False),
                nn.BatchNorm2d(output),)


    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        x = self.downsample(x)
        return F.relu(x.add(y),True)


class Residual9(nn.Module):#(64,128)
    def __init__(self, input, output):
        super(Residual9,self).__init__()

        self.conv1 = nn.Conv2d(input, output, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(output,output,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output)
         
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)       
        y = self.conv2(y)
        y = self.bn2(y)
        #x = self.downsample(x)
        return F.relu(x.add(y),True)
    
class Net(nn.Module):
    
    def __init__(self, num_classes=625,get_features=False):
        super(Net,self).__init__()
        # 3 128 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),                       
        )
        self.conv2 = nn.Sequential(           
            nn.Conv2d(32,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )       
        self.maxpool3=nn.MaxPool2d(3,2,padding=1)# ceil_mode=True
        # 32 64 32     
        self.residual4 = Residual4(32,32)                            
        self.residual5 = Residual5(32,32)
        # 32 64 32
        self.residual6=Residual6(32,64)
        self.residual7=Residual7(64,64)
        # 64 32 16
        self.residual8=Residual8(64,128)
        self.residual9=Residual9(128,128)
# =============================================================================
#         self.dense = nn.Sequential(
#             nn.Dropout(p=0.6),
#             nn.Linear(128*16*8, 128),)    
#         # 128 16 8  
# =============================================================================
        self.get_features = get_features
        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(128*16*8, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True)
        )
        
        self.batch_norm = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool3(x)
        x = self.residual4(x)
        x = self.residual5(x)
        x = self.residual6(x)
        x = self.residual7(x)
        x = self.residual8(x)
        x = self.residual9(x)
             
        #############################################
        #print(x.shape)#torch.Size([2, 128, 16, 8])
        x=x.flatten(1)
        #print(x.shape)#torch.Size([2, 16384])
        if self.get_features:
            x = self.dense[0](x)
            x = self.dense[1](x)

            #2,128 
            x = x.div(x.norm(p=2,dim=1,keepdim=True))                   
            #x= x.div(torch.sqrt(torch.sum(x*x,dim=1,keepdim=True) ))
            
            #y=x.norm(p=2,dim=1,keepdim=True)
            #x = x.div(y)
            #x = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-12)
            #y=F.normalize(x, p=2, dim=1)
            #x = x / (y + 1e-12)
            #print(x.shape)
            return x
        
        x = self.dense(x)
        x = x.div(x.norm(p=2,dim=1,keepdim=True))#reducel2
        #x= x.div(torch.sqrt(torch.sum(x*x,dim=1,keepdim=True) ))#reducesum
        #x = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-12)
        #y=F.normalize(x, p=2, dim=1)
        #x = x / (y + 1e-12)
        #x = F.normalize(x, p=2, dim=1)
        x = self.classifier(x)
        return x

       
if __name__ == '__main__':
    setup_seed(10)
    x = torch.randn(2,3,128,64)
    net=Net(get_features=True)
    print(net)
    print(net(x))
  
    
    
    
    #import ipdb; ipdb.set_trace()


