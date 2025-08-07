''' Define the sublayers in encoder/decoder layer 
Derived From : https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    '''Scaled Dot-Product Attention 
    '''

    def __init__(self, temperature):
        '''
        Initialize the model.
        :param temperature: TODO ....
        :param attn_dropout: Argument to dropout after softmax(QK)
        '''
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        '''
        Callback Function:
        :param q: The Query matrix.
        :param k: The Key matrix.
        :param v: The value matrix.
        :param mask: The mask of the input.
        :returns (output, attention): A tuple consisting of softmax(QK^T)V and softmax(QK^T)
        '''
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output

        
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v):
        '''
        Intialize the model.
        :param n_head: Number of self-attention modules
        :param d_model: Dimension of input/output of this layer
        :param d_k: Dimension of each Key
        :param d_v: Dimension of each Value
        :param dropout: 
        '''
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        '''
        Callback function.
        :param q: The Query matrix.
        :param k: The Key matrix.
        :param v: The value matrix.
        :param mask: The mask to use.
        :returns (output, attention): A tuple consisting of network output and softmax(QK^T)
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b_q = q.size(0)
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b_q, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q


class PositionwiseFeedForward(nn.Module):
    ''' A simple 2 layer with fully connected layer.
    '''

    def __init__(self, d_in, d_hid):
        '''
        Initialize the model.
        :param d_in: Dimension of the input/output of the model.
        :param d_hid: Dimension of the hidden layer.
        :param dropout: Argument to the dropout layer.
        '''
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):
        '''
        Callback function.
        :param x: The input to the function.
        :returns torch.array: An output of the same dimension as the input.
        '''
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x += residual

        x = self.layer_norm(x)

        return x