import torch
import torch.nn as nn
import torchvision

class BuildingBlock(nn.Module):
    """
    Blocks for Resnet18 and Resnet 34.
    
    """
    mul = 1
    def __init__(self, input_nc, output_nc, stride=1, downsample=None):
        super(BuildingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, output_nc, kernel_size=3,stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(output_nc)
        
        self.conv2 = nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_nc)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x =  x + identity
        x = self.relu(x)

        return x
        
        
        
class BottleneckBlock(nn.Module):
    """
    Blocks for Resnet 50, 101, 152
    """
    mul = 4
    def __init__(self, input_nc, output_nc, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, output_nc, kernel_size=1,stride=stride)
        self.bn1 = nn.BatchNorm2d(output_nc)
        
        self.conv2 = nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_nc)
        
        self.conv3 = nn.Conv2d(output_nc, output_nc * self.mul, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(output_nc * self.mul)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x= self.conv3(x)
        x= self.bn3(x)
        x =  x + identity
        x = self.relu(x)        
        
        return x
    
def _downsample(input_nc, output_nc,stride=2):
    """
    downsampling when activation map size does not match
    """
    down = []
    down.append(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=stride))
    down.append(nn.BatchNorm2d(output_nc))
    return nn.Sequential(*down)

def make_layer(block, num_layer, input_nc, output_nc, stride=2):
    """
    make layer conv2 to conv5
    """
    layers = []
    downsample = None
    for i in range(num_layer):
        if stride == 2 or input_nc != output_nc * block.mul:
            downsample = _downsample(input_nc, output_nc*block.mul, stride)
        layers.append(block(input_nc, output_nc,stride, downsample))
        
        input_nc = output_nc * block.mul
        
        stride = 1
        downsample = None
        
        
    return nn.Sequential(*layers)

class Resnet(nn.Module):
    def __init__(self,block=BottleneckBlock,layers=[3,4,6,3], num_classes=1000):
        super(Resnet, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(3,stride=2, padding=1))
          
        #b building block
        self.conv2 = make_layer(block, layers[0], 64,64, stride=1)
        self.conv3 = make_layer(block, layers[1],64*block.mul,128)
        self.conv4 = make_layer(block, layers[2], 128*block.mul,256)
        self.conv5 = make_layer(block, layers[3], 256*block.mul, 512)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.mul ,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    
def resnet18(num_classes=1000):
    model = Resnet(block=BuildingBlock,layers=[2,2,2,2], num_classes=num_classes)
    return model
def resnet34(num_classes=1000):
    model = Resnet(block=BuildingBlock,layers=[3,4,6,3], num_classes=num_classes)
    return model

def resnet50(num_classes=1000):
    model = Resnet(block=BottleneckBlock, layers=[3,4,6,3], num_classes=num_classes)
    return model

def resnet101(num_classes=1000):
    model = Resnet(block=BottleneckBlock, layers=[3,4,23,3],num_classes=num_classes)
    return model

def resnet152(num_classes=1000):
    model = Resnet(block=BottleneckBlock, layers=[3,8,36,3],num_classes=num_classes)
    return model