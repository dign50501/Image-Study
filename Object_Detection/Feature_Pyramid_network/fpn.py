import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101

def conv1x1(input_nc, output_nc):
    return nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0)

def conv3x3():
    return nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

class FPN(nn.Module):
    def __init__(self,resnet='resnet50'):
        super(FPN, self).__init__()
        if resnet == 'resnet50':
            self.resnet = resnet50(pretrained=True)
        elif resnet == 'resnet101':
            self.resnet = resnet101(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.conv2 = self.resnet.layer1
        self.conv3 = self.resnet.layer2
        self.conv4 = self.resnet.layer3
        self.conv5 = self.resnet.layer4
        
        self.top_conv = conv1x1(2048, 256)
        self.c4_conv = conv1x1(1024, 256)
        self.c3_conv = conv1x1(512, 256)
        self.c2_conv = conv1x1(256, 256)
        
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.p4_conv = conv3x3()
        self.p3_conv = conv3x3()
        self.p2_conv = conv3x3()

    def forward(self, x):
        # bottom-up pathway
        C1 = self.conv1(x)
        C2 = self.conv2(C1) # stride of 4  (224 / 4 = 56) C2.shape=[batch, 256, 56, 56] 
        C3 = self.conv3(C2) # stride of 8  (224 / 8 = 28) C3.shape=[batch, 512, 28 ,28]
        C4 = self.conv4(C3) # stride of 16 (224 / 16 = 14) C4.shape=[batch, 1024, 14, 14]
        C5 = self.conv5(C4) # stride of 32 (224 / 32 = 7) C5.shape=[batch, 2048, 28, 28]
        
        # top-down pathway
        P5 = self.top_conv(C5)
        
        
        P4 = self.p4_conv(self.upsampling(P5) + self.c4_conv(C4))
        P3 = self.p3_conv(self.upsampling(P4) + self.c3_conv(C3))
        P2 = self.p2_conv(self.upsampling(P3) + self.c2_conv(C2))
        return P2, P3, P4, P5

if __name__ == '__main__':
    x = torch.rand([2,3,224,224])
    # check if size is correct
    fpn = FPN('resnet101')
    P2, P3,P4, P5 = fpn(x)
    print(P2.shape) 
    print(P3.shape) 
    print(P4.shape) 
    print(P5.shape) 