
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import timm
def filter(opState, kernelsize=5):
    #B*100*2

    Bs = opState.shape[0]
    ches = opState.shape[1]
    recL = int((kernelsize-3)/2)
    labelTable = torch.zeros(Bs, int(ches+2*recL),opState.shape[2]).cuda()
    labelTable[:,:recL,:] =   opState[:,0,:].unsqueeze(dim=1)
    labelTable[:,-recL:,:] =   opState[:,-1,:].unsqueeze(dim=1)
    labelTable[:,recL:-recL,:] = opState

    newOpState = torch.zeros_like(opState)

    tmpT = labelTable.unfold(1, kernelsize, 1)
    tmpMeanT = torch.mean(tmpT, dim=-1)
    newOpState[:,1:-1,:] = tmpMeanT


    newOpState[:,0,:] = opState[:,0,:]
    newOpState[:,-1,:] = opState[:,-1,:]
    

    return newOpState
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=600):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


class resnet101(nn.Module):
    def __init__(self,  filter = 7):
        super(resnet101, self).__init__()
        self.image = timm.create_model('resnet101',num_classes=600)
        self.image._modules['conv1'] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # print(self.image)
        self.filter = filter

    def forward(self, x, labelState, labelRot):
        Bs = x.shape[0]
        ft = torch.reshape(self.image(x),(Bs,200,3))
        # print("lo.shape", lo.shape)
        opState = torch.zeros_like(labelState) 
        yw = ft[:,:,2]
        cosyaw = torch.cos(yw).unsqueeze(dim=2) #b*100*1
        sinyaw = torch.sin(yw).unsqueeze(dim=2) #b*100*1
        rotOutput = torch.cat((cosyaw,sinyaw), dim=2) #b*100*2
        opState[:,1:-1,:] = ft[:,1:-1,0:2]
        opState[:,0,:] = labelState[:,0,:]
        opState[:,-1,:] = labelState[:,-1,:]
        # print("rotOutput.shape", rotOutput.shape)
        # print("labelRot.shape", labelRot.shape)
        rotOutput[:,0,:] = labelRot[:,0,:]
        rotOutput[:,-1,:] = labelRot[:,-1,:]
        if(self.filter >=3):
            opState = filter(opState, self.filter)
            rotOutput = filter(rotOutput, self.filter)#B*99*2
            rotOutput = torch.nn.functional.normalize(rotOutput, dim=2)


        return opState, rotOutput
    
class resnet50(nn.Module):
    def __init__(self,  filter = 7):
        super(resnet50, self).__init__()
        self.image = timm.create_model('resnet50',num_classes=600)
        self.image._modules['conv1'] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # print(self.image)
        self.filter = filter

    def forward(self, x, labelState, labelRot):
        Bs = x.shape[0]
        ft = torch.reshape(self.image(x),(Bs,200,3))
        # print("lo.shape", lo.shape)
        opState = torch.zeros_like(labelState) 
        yw = ft[:,:,2]
        cosyaw = torch.cos(yw).unsqueeze(dim=2) #b*100*1
        sinyaw = torch.sin(yw).unsqueeze(dim=2) #b*100*1
        rotOutput = torch.cat((cosyaw,sinyaw), dim=2) #b*100*2
        opState[:,1:-1,:] = ft[:,1:-1,0:2]
        opState[:,0,:] = labelState[:,0,:]
        opState[:,-1,:] = labelState[:,-1,:]
        # print("rotOutput.shape", rotOutput.shape)
        # print("labelRot.shape", labelRot.shape)
        rotOutput[:,0,:] = labelRot[:,0,:]
        rotOutput[:,-1,:] = labelRot[:,-1,:]
        if(self.filter >=3):
            opState = filter(opState, self.filter)
            rotOutput = filter(rotOutput, self.filter)#B*99*2
            rotOutput = torch.nn.functional.normalize(rotOutput, dim=2)


        return opState, rotOutput
    
class resnet152(nn.Module):
    def __init__(self, filter = 7):
        super(resnet152, self).__init__()
        self.image = timm.create_model('resnet152',num_classes=600)
        self.image._modules['conv1'] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # print(self.image)
        self.filter = filter

    def forward(self, x, labelState, labelRot):
        Bs = x.shape[0]
        ft = torch.reshape(self.image(x),(Bs,200,3))
        # print("lo.shape", lo.shape)
        opState = torch.zeros_like(labelState) 
        yw = ft[:,:,2]
        cosyaw = torch.cos(yw).unsqueeze(dim=2) #b*100*1
        sinyaw = torch.sin(yw).unsqueeze(dim=2) #b*100*1
        rotOutput = torch.cat((cosyaw,sinyaw), dim=2) #b*100*2
        opState[:,1:-1,:] = ft[:,1:-1,0:2]
        opState[:,0,:] = labelState[:,0,:]
        opState[:,-1,:] = labelState[:,-1,:]
        # print("rotOutput.shape", rotOutput.shape)
        # print("labelRot.shape", labelRot.shape)
        rotOutput[:,0,:] = labelRot[:,0,:]
        rotOutput[:,-1,:] = labelRot[:,-1,:]
        if(self.filter >=3):
            opState = filter(opState, self.filter)
            rotOutput = filter(rotOutput, self.filter)#B*99*2
            rotOutput = torch.nn.functional.normalize(rotOutput, dim=2)


        return opState, rotOutput
    
  



# test()
if __name__ == "__main__":
    # test()
    pt = 200
    model = resnet152(pt_num=pt, filter=7).cuda().half()
    totalt = 0.0
    count = 0
    model.eval()
    # out = model(input)
    input  = torch.rand(1,4,200,200).cuda().half()
    labelState = torch.rand(1,pt,2).cuda().half()
    labelRot = torch.rand(1,pt,2).cuda().half()
    anchors = torch.rand(1,pt,20,20).cuda().half()
    with torch.no_grad():
        for i in range(100):
            torch.cuda.synchronize()
            start = time.time()
            out = model(input, labelState, labelRot, anchors)
            torch.cuda.synchronize()
            end = time.time()
            if i>=10:
                totalt += 1000.0*(end-start)
                count +=1
    print("model time: ", totalt / count, " ms")
    
    