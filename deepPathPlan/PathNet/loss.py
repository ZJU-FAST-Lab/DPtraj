
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
    
def positiveSmoothedL1(x):
    #x:B*99
    pe = 1.0e-4
    half = 0.5 * pe
    f3c = 1.0 / (pe * pe)
    f4c = -0.5 * f3c / pe

    b1 = x <= 0.0
    b2 = (x>0.0) & (x < pe)
    b3 = x >= pe

    a1 = 0.0
    a2 = (f4c * x + f3c) * x * x * x
    a3 = x - half
    loss = a1 * b1 + a2 * b2 + a3 * b3

    return loss

def positiveSmoothedL2(x):
    #x:B*99
    f = nn.ReLU()
    x = f(x)
    loss  = x * x
    return loss

def positiveSmoothedL3(x):
    #x:B*99
    f = nn.ReLU()
    x = f(x)
    loss  = x * x *x
    return loss

def floatToInt(pos, res=0.1):
    #pos B*C*2
    gridIdx = Variable(((pos+10.0)/res).floor().long())
    return gridIdx

def intToFloat(grid, res=0.1):
    pos = Variable(((grid+0.5)*res-10.0).float())
    return pos

def getDistGrad(inputpos, esdfMap, res=0.1):
    #pos B*C*2
    #esdfMap B*3*200*200
    outRange=  (inputpos < -9.9) | (inputpos > 9.9)
    outRange = outRange[:,:,0] | outRange[:,:,1]
    # print(outRange)
    pos = torch.clamp(inputpos,-9.9,9.9)
    # pos = inputpos
    pos_m = pos - 0.5 * res
    idx = floatToInt(pos_m) #B*C*2
    idx_pos = intToFloat(idx)
    diff = (pos - idx_pos) / res #B*C*2
    
    Bs = pos.shape[0]
    channels = pos.shape[1]
    values = torch.zeros(Bs,channels, 2,2).cuda()
    for x in range(0,2):
        for y in range(0,2):
            offset = Variable(torch.tensor([x,y]).long().cuda())
            current_idx = idx + offset#B*C*2
            for i in range(Bs):
                values[i,:,x,y] = esdfMap[i,0,current_idx[i,:,0], current_idx[i,:,1]]
    values = Variable(values)
    v00 = (1-diff[:,:,0]) * values[:,:,0,0] + diff[:,:,0] * values[:,:,1,0]
    v10 = (1-diff[:,:,0]) * values[:,:,0,1] + diff[:,:,0] * values[:,:,1,1]
    v0 = (1-diff[:,:,1]) * v00 + diff[:,:,1] * v10
    v0 = torch.where(outRange, -1.0, v0)
    return v0

def getSqureArc(inputs):
    pts1 = inputs[:,:-1,:]
    pts2 = inputs[:,1:,:]
    dif = pts1 - pts2
    dif = dif * dif
    arc = torch.sum(dif, dim=2)
    arc = torch.sum(arc, dim=1)
    return arc
    
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.95, gamma=1.0, smooth=1):       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
             
        return loss.mean()
    
class ArcLoss(nn.Module):
    def __init__(self):
        super(ArcLoss, self).__init__()

    def forward(self, inputs):
        pts1 = inputs[:,:-1,:]
        pts2 = inputs[:,1:,:]
        arcs = pts1 - pts2
        arcs = torch.norm(arcs, dim=2)
        loss = torch.sum(arcs, dim=1)
        return loss
    
class NormalizeArcLoss(nn.Module):
    def __init__(self):
        super(NormalizeArcLoss, self).__init__()
        self.arcloss = ArcLoss()

    def forward(self, inputs, labels):
        arc1 = self.arcloss(inputs)
        arc2 = self.arcloss(labels)
        vioarc = arc1-arc2
        arcloss = positiveSmoothedL1(vioarc)
        return torch.mean(arcloss)

class NormalizeRotLoss(nn.Module):
    def __init__(self):
        super(NormalizeRotLoss, self).__init__()

    def forward(self, inputs, labels):
        dif1 = getSqureArc(inputs)
        dif2 = getSqureArc(labels)
        normlizeArc = dif1 / dif2 #B
        return torch.mean(normlizeArc)

class SmoothTrajLoss(nn.Module):
    def __init__(self):
        super(SmoothTrajLoss, self).__init__()
    def forward(self, opState):
        #B*C*2
        pts1 = opState[:,:-1,:]  
        pts2 = opState[:,1:,:]
        v = pts2-pts1
        normV = torch.nn.functional.normalize(v, dim=2)#B 99 2
        v1 = normV[:,:-1,:]
        v2 = normV[:,1: ,:]
        c = torch.zeros_like(v2)#B 98 2
        c[:,:,0] = -v2[:,:,1]
        c[:,:,1] =  v2[:,:,0]
        cross = torch.sum(v1 * c,dim=2)#B*98
        cross = torch.pow(cross,2)#b 98
        loss = torch.sum(cross, dim=1)#B 
        return loss

class NormalizeSmoothTrajLoss(nn.Module):
    def __init__(self):
        super(NormalizeSmoothTrajLoss, self).__init__()
        self.s = SmoothTrajLoss()
    def forward(self, inputs, labels):
        loss1 = self.s(inputs)
        loss2 = self.s(labels)
        vios = loss1-loss2
        sloss = positiveSmoothedL1(vios) #B
        return torch.mean(sloss)

class GearTrajLoss(nn.Module):
    def __init__(self):
        super(GearTrajLoss, self).__init__()
    def forward(self, opState):
        #B*C*2
        pts1 = opState[:,:-1,:]  
        pts2 = opState[:,1:,:]
        v = pts2-pts1
        normV = torch.nn.functional.normalize(v, dim=2)#B 99 2
        v1 = normV[:,:-1,:]
        v2 = normV[:,1: ,:]
        dif = v2-v1 #B*98*2
        dfi2 = torch.pow(dif,2) # B*98*2
        loss = torch.sum(dfi2, dim=2)#B*98
        loss = torch.sum(loss, dim=1)#B
        return loss

class NormalizeGearTrajLoss(nn.Module):
    def __init__(self):
        super(NormalizeGearTrajLoss, self).__init__()
        self.s = GearTrajLoss()
    def forward(self, inputs, labels):
        loss1 = self.s(inputs)
        loss2 = self.s(labels)
        loss = loss1/loss2
        
        vios = 8.0*(loss-1.0)#B
        sloss = positiveSmoothedL1(vios) #B
        return torch.mean(sloss)

class UniforArcLoss(nn.Module):
    def __init__(self):
        super(UniforArcLoss, self).__init__()
        self.arcloss = ArcLoss()

    def forward(self, inputs, labels):
        pts1 = inputs[:,:-1,:]  
        pts2 = inputs[:,1:,:]
        arcs = pts1 - pts2 #B*99*2
        arcs = torch.norm(arcs, dim=2)#B*99
        varloss = torch.std(arcs, dim=1, unbiased=False) #B
        labelArc = self.arcloss(labels) #B
        normalizeLoss = varloss / labelArc
        loss = torch.mean(normalizeLoss)
        return loss

class NonholoLoss(nn.Module):
    def __init__(self):
        super(NonholoLoss, self).__init__()

    def forward(self, inputs, labels):
        #labels cos sin
        pts1 = inputs[:,:-1,:]  
        pts2 = inputs[:,1:,:]
        arcs = pts2 - pts1
        arcs = torch.nn.functional.normalize(arcs, dim=2)
        
        labeldir = torch.zeros_like(labels)
        labeldir[:,:,0] = -labels[:,:,1]
        labeldir[:,:,1] =  labels[:,:,0]
        
        cross = torch.sum(arcs * labeldir,dim=2)
        cross = torch.pow(cross,2)
        
        
        vioh = (cross-0.067)
        hloss = positiveSmoothedL1(vioh)
        return torch.mean(hloss)
     
class CurvatureLoss(nn.Module):
    def __init__(self):
        super(CurvatureLoss, self).__init__()
        self.kmax = 1.67
    def forward(self, opstate, rot):
        rot1 = rot[:, :-1, :]
        rot2 = rot[:, 1:, :]
        rotarc = rot2-rot1
        deltaAngles = torch.norm(rotarc,dim=2)
        
        pts1 = opstate[:,:-1,:]  
        pts2 = opstate[:,1:,:]
        arcs = pts2 - pts1
        arcs = torch.norm(arcs, dim=2)
        viok = (deltaAngles * deltaAngles - self.kmax * self.kmax * (arcs + 1.0e-3) * (arcs + 1.0e-3))
        kloss = positiveSmoothedL1(viok)
        return torch.mean(kloss)
    
class TurnLoss(nn.Module):
    def __init__(self):
        super(TurnLoss, self).__init__()
    def forward(self, opState):
        pts1 = opState[:,:-1,:]  
        pts2 = opState[:,1:,:]
        v = pts2-pts1
        normV = torch.nn.functional.normalize(v, dim=2)
        v1 = normV[:,:-1,:]
        v2 = normV[:,1: ,:]
        c = torch.zeros_like(v2)
        c[:,:,0] = -v2[:,:,1]
        c[:,:,1] =  v2[:,:,0]
        cross = torch.sum(v1 * c,dim=2)

        cross = torch.pow(cross,2)
        vio = (cross - 0.35)
        sloss = positiveSmoothedL1(vio)
        return torch.mean(sloss)

class CollisionLoss(nn.Module):
    def __init__(self):
        super(CollisionLoss, self).__init__()
    def forward(self, opState, envs):
        #inputs:B*C*2       B*3*200*200
        dists = getDistGrad(opState, envs)
        viod = 10.0*(0.3-dists)
        penalty = positiveSmoothedL2(viod)
        return torch.mean(penalty)
    
class FullShapeCollisionLoss(nn.Module):
    def __init__(self):
        super(FullShapeCollisionLoss, self).__init__()
        self.conpts = torch.tensor([[0.15, 0.0], [0.45, -0.00]]).cuda()#10*2
    def forward(self, opState, opRot, envs):
        #inputs:B*C*2       B*3*200*200
        B = opState.shape[0]
        C = opState.shape[1]
        N = self.conpts.shape[0]
        rotR = torch.zeros(B,C,2,2).cuda()
        cos = opRot[:,:,0]
        sin = opRot[:,:,1]
        rotR[:,:,0,0] = cos
        rotR[:,:,0,1] = -sin
        rotR[:,:,1,0] = sin
        rotR[:,:,1,1] = cos
        offset = torch.einsum('bcij,nj->bcni',rotR, self.conpts)
        absPt = opState.unsqueeze(dim=2) # B C 1 2
        absPt = absPt.repeat(1,1,N,1)
        absPt = absPt + offset #B C N 2
        absPt = rearrange(absPt, 'b c n i-> b (c n) i') #B*CN*2
        dists = getDistGrad(absPt, envs)
        viod = 10.0*(0.3-dists)
        penalty = positiveSmoothedL1(viod)
        return torch.mean(penalty)
    
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.wei_arc = torch.tensor(0.5)
        self.wei_uni = torch.tensor(100.0)
        self.wei_hol = torch.tensor(500.0)
        self.wei_cur = torch.tensor(500.0)
        self.wei_safety = torch.tensor(500.0)
        self.wei_rsm = torch.tensor(0.5)
        self.wei_traj = torch.tensor(0.3)
        self.wei_turn  = torch.tensor(100.0)
        
        self.smooLoss = NormalizeArcLoss()
        self.rotsLoss = NormalizeArcLoss()
        self.holoLoss = NonholoLoss()
        self.uniLoss = UniforArcLoss()
        self.curLoss = CurvatureLoss()
        self.safeLoss = FullShapeCollisionLoss()
        self.trajLoss = NormalizeSmoothTrajLoss()
        self.turnLoss = TurnLoss()

    def forward(self, opState, label_opState, opRot, label_opRot,envs):
        arcloss = self.wei_arc * self.smooLoss(opState, label_opState)
        holloss = self.wei_hol * self.holoLoss(opState,opRot[:,1:,:])
        unilos = self.wei_uni * self.uniLoss(opState,label_opState)
        curloss = self.wei_cur * self.curLoss(opState, opRot)
        rsmloss = self.wei_rsm * self.rotsLoss(opRot, label_opRot)
        safetyloss = self.wei_safety * self.safeLoss(opState, opRot, envs)
        trajloss = self.wei_traj * self.trajLoss(opState, label_opState)
        turnloss = self.wei_turn * self.turnLoss(opState)
        loss = arcloss + holloss + unilos  + curloss + rsmloss + safetyloss+trajloss+turnloss
        return loss