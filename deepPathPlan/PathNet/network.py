import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
import numpy as np
from Layers import EncoderLayer
from einops.layers.torch import Rearrange

def filter(opState, kernelsize=5):
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

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
    
    
class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PositionalEncoding(nn.Module):
    '''Positional encoding
    '''
    def __init__(self, d_hid, n_position, train_shape):
        '''
        Intialize the Encoder.
        :param d_hid: Dimesion of the attention features.
        :param n_position: Number of positions to consider.
        :param train_shape: The 2D shape of the training model.
        '''
        super(PositionalEncoding, self).__init__()
        self.n_pos_sqrt = int(np.sqrt(n_position))
        self.train_shape = train_shape
        # Not a parameter
        self.register_buffer('hashIndex', self._get_hash_table(n_position))
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.register_buffer('pos_table_train', self._get_sinusoid_encoding_table_train(n_position, train_shape))

    def _get_hash_table(self, n_position):
        '''
        A simple table converting 1D indexes to 2D grid.
        :param n_position: The number of positions on the grid.
        ''' 

        return rearrange(torch.arange(n_position), '(h w) -> h w', h=int(np.sqrt(n_position)), w=int(np.sqrt(n_position))) # 40 * 40

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        '''
        Sinusoid position encoding table.
        :param n_position:
        :param d_hid:
        :returns 
        '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table[None,:])
    
    def _get_sinusoid_encoding_table_train(self, n_position, train_shape):
        '''
        The encoding table to use for training.
        NOTE: It is assumed that all training data comes from a fixed map.
        NOTE: Another assumption that is made is that the training maps are square.
        :param n_position: The maximum number of positions on the table.
        :param train_shape: The 2D dimension of the training maps.
        '''
        selectIndex = rearrange(self.hashIndex[:train_shape[0], :train_shape[1]], 'h w -> (h w)') # 24 * 24
        return torch.index_select(self.pos_table, dim=1, index=selectIndex)

    def forward(self, x, conv_shape=None):
        '''
        Callback function
        :param x:
        '''
        if conv_shape is None:
            startH, startW = torch.randint(0, self.n_pos_sqrt-self.train_shape[0], (2,))
            selectIndex = rearrange(
                self.hashIndex[startH:startH+self.train_shape[0], startW:startW+self.train_shape[1]],
                'h w -> (h w)'
                )
            return x + torch.index_select(self.pos_table, dim=1, index=selectIndex).clone().detach()

        # assert x.shape[0]==1, "Only valid for testing single image sizes"
        selectIndex = rearrange(self.hashIndex[:conv_shape[0], :conv_shape[1]], 'h w -> (h w)')
        return x + self.pos_table[:, selectIndex.long(), :]


class Encoder(nn.Module):
    ''' The encoder of the planner.
    '''

    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, n_position, train_shape):
        '''
        Intialize the encoder.
        :param n_layers: Number of layers of attention and fully connected layer.
        :param n_heads: Number of self attention modules.
        :param d_k: Dimension of each Key.
        :param d_v: Dimension of each Value.
        :param d_model: Dimension of input/output of encoder layer.
        :param d_inner: Dimension of the hidden layers of position wise FFN
        :param pad_idx: TODO ....
        :param dropout: The value to the dropout argument.
        :param n_position: Total number of patches the model can handle.
        :param train_shape: The shape of the output of the patch encodings.
        '''
        super().__init__()
        self.to_patch_embedding = nn.Sequential(

            
            (DoubleConv(4, 64)),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.reorder_dims = Rearrange('b c h w -> b (h w) c')
        # Position Encoding.
        # NOTE: Current setup for adding position encoding after patch Embedding.
        self.position_enc = PositionalEncoding(d_model, n_position=n_position, train_shape=train_shape)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_heads, d_k, d_v)
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        

    def forward(self, input_map, returns_attns=False):
        '''
        The input of the Encoder should be of dim (b, c, h, w).
        :param input_map: The input map for planning.
        :param returns_attns: If True, the model returns slf_attns at each layer
        '''
        enc_slf_attn_list = []
        enc_output = self.to_patch_embedding(input_map)
        conv_map_shape = enc_output.shape[-2:]
        enc_output = self.reorder_dims(enc_output)

        if self.training:
            enc_output = self.position_enc(enc_output)
        else:
            enc_output = self.position_enc(enc_output, conv_map_shape)
        
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=None)
        
        if returns_attns:
            return enc_output, enc_slf_attn_list
        return enc_output, 
    
    
class Transformer(nn.Module):
    ''' A Transformer module
    '''
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, n_position, train_shape):
        '''
        Initialize the Transformer model.
        :param n_layers: Number of layers of attention and fully connected layers
        :param n_heads: Number of self attention modules.
        :param d_k: Dimension of each Key.
        :param d_v: Dimension of each Value.
        :param d_model: Dimension of input/output of decoder layer.
        :param d_inner: Dimension of the hidden layers of position wise FFN1
        :param pad_idx: TODO ......
        :param dropout: The value of the dropout argument.
        :param n_position: Dim*dim of the maximum map size.
        :param train_shape: The shape of the output of the patch encodings. 
        '''
        super().__init__()

        self.encoder = Encoder(
            n_layers=n_layers, # num of sublayer
            n_heads=n_heads,  # a dimension in query, key, value
            d_k=d_k, # dimension of key
            d_v=d_v, # dimension of value
            d_model=d_model, # channel of conv as a first part
            d_inner=d_inner, # channel of inner part in the model
            pad_idx=pad_idx, 
            n_position=n_position, # max table size for position encoding
            train_shape=train_shape # image size in meters
        )

    def forward(self, input_map):
        '''
        The callback function.
        :param input_map:
        :param goal: A 2D torch array representing the goal.
        :param start: A 2D torch array representing the start.
        :param cur_index: The current anchor point of patch.
        '''
        enc_output, *_ = self.encoder(input_map)
        enc_output = rearrange(enc_output, 'b c d -> b d c')
        enc_output = rearrange(enc_output, 'b c (h w) -> b c h w', h = 20)
        return enc_output

class AnchorNet25(nn.Module):
    def __init__(self, n_channels, out_channels=1):
        super(AnchorNet25, self).__init__()
        model_args = dict(
            n_layers=6, 
            n_heads=3, 
            d_k=512, 
            d_v=256, 
            d_model=512, 
            d_inner=1024, 
            pad_idx=None,
            n_position=40*40, 
            # train_shape=[25, 25],
            train_shape=[20, 20]
        )
        self.transformer = Transformer(**model_args)


        self.outc = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            OutConv(1024, out_channels)
        )

    def forward(self, x):
        x = self.transformer(x)
        result = self.outc(x)
        result_1 = rearrange(result, 'b c h w-> b c (h w)')
        result_2 = rearrange(result_1, 'b c l-> (b c) l')
        result = rearrange(result, 'b c h w-> b c (h w)')
        result = torch.softmax(result, dim=2)
        result = rearrange(result, 'b c (h w)-> b c h w', h = 20)
        return x,result,result_2


class trajFCNet(nn.Module):
    def __init__(self, image_channels=4, pt_num=100, filter = 7, l = 1.2, use_groundTruth = True):
        super(trajFCNet, self).__init__()
        self.a = AnchorNet25(image_channels, pt_num)
    
        self.fcntrajout = nn.Sequential(
            nn.Conv2d(512+pt_num, 1024, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            OutConv(1024, pt_num*3)
        )
     

        self.pt_num = pt_num
        self.filter = filter
        self.l = l
        self.w = (self.l -1.0)/2.0

        self.use_groundTruth = use_groundTruth
        self.sig = nn.Sigmoid()#0~1
        xylim = torch.arange(0,20).unsqueeze(dim=1)
        xylim = xylim.repeat(1,20)
        yxlim = torch.arange(0,20).unsqueeze(dim=0)
        yxlim = yxlim.repeat(20,1)
        self.register_buffer('xylim', xylim)
        self.register_buffer('yxlim', yxlim)




    def forward(self, x, labelState, labelRot, anchors):
        ft, result_1, result_2 = self.a(x)
        
        
        prbmap = torch.zeros_like(result_1)
        prbmap[:,0,:,:] = anchors[:,0,:,:]
        prbmap[:,-1,:,:] = anchors[:,-1,:,:]
        prbmap[:,1:-1,:,:] = result_1[:,1:-1,:,:]


        resFeature= ft
        if(self.use_groundTruth):
            anchorsFeature = anchors
        else:
            anchorsFeature = prbmap

        resInput = torch.cat((anchorsFeature, resFeature), dim=1)

        resOutput = self.fcntrajout(resInput)
        if(self.use_groundTruth):
            if(self.l>0):
                #hzchzc
                px = (self.l*self.sig(resOutput[:,0::3,:,:])-self.w)* anchors 
                py = (self.l*self.sig(resOutput[:,1::3,:,:])-self.w)* anchors
            else:
                px = resOutput[:,0::3,:,:]* anchors
                py = resOutput[:,1::3,:,:]* anchors
            yw =               resOutput[:,2::3,:,:]  * anchors
        else:
            if(self.l>0):
                px = (self.l*self.sig(resOutput[:,0::3,:,:])-self.w)* prbmap
                py = (self.l*self.sig(resOutput[:,1::3,:,:])-self.w)* prbmap
            else:
                px = resOutput[:,0::3,:,:]* prbmap 
                py = resOutput[:,1::3,:,:]* prbmap
            yw =               resOutput[:,2::3,:,:]  * prbmap
        
        # bias
        px = torch.sum(torch.sum(px, dim = 3), dim=2).unsqueeze(dim=2)
        py = torch.sum(torch.sum(py, dim = 3), dim=2).unsqueeze(dim=2)
        yw = torch.sum(torch.sum(yw, dim = 3), dim=2)

        gridx = Variable(self.xylim, requires_grad = False)
        gridy = Variable(self.yxlim, requires_grad = False)
        if(self.use_groundTruth):
            xmap = anchors * gridx
            ymap = anchors * gridy
        else:    
            xmap = prbmap * gridx
            ymap = prbmap * gridy
        aveGirdx = torch.sum(torch.sum(xmap, dim=3), dim=2).unsqueeze(dim=2)
        aveGirdy = torch.sum(torch.sum(ymap, dim=3), dim=2).unsqueeze(dim=2)
        # local origin
        lo = torch.cat((aveGirdx, aveGirdy), dim=2)*1.0-10.0
        opState = torch.cat((px, py), dim=2) + lo
        cosyaw = torch.cos(yw).unsqueeze(dim=2)
        sinyaw = torch.sin(yw).unsqueeze(dim=2)
        rotOutput = torch.cat((cosyaw,sinyaw), dim=2)
        opState[:,0,:] = labelState[:,0,:]
        opState[:,-1,:] = labelState[:,-1,:]
        rotOutput[:,0,:] = labelRot[:,0,:]
        rotOutput[:,-1,:] = labelRot[:,-1,:]
        if(self.filter >=3):
            opState = filter(opState, self.filter)
            rotOutput = filter(rotOutput, self.filter)
            rotOutput = torch.nn.functional.normalize(rotOutput, dim=2)


        return opState, rotOutput, prbmap, result_2

if __name__ == "__main__":
    pt = 200
    model = trajFCNet(pt_num=pt, filter=7).cuda().half()
    totalt = 0.0
    count = 0
    model.eval()
    input  = torch.rand(1,4,200,200).cuda().half()
    labelState = torch.rand(1,pt,2).cuda().half()
    labelRot = torch.rand(1,pt,2).cuda().half()
    anchors = torch.rand(1,pt,20,20).cuda().half()
    out = model(input, labelState, labelRot, anchors)
    with torch.no_grad():
        for i in range(200):
            torch.cuda.synchronize()
            start = time.time()
            out = model(input, labelState, labelRot, anchors)
            torch.cuda.synchronize()
            end = time.time()
            if i>=20:
                totalt += 1000.0*(end-start)
                count +=1
    print("model time: ", totalt / count, " ms")