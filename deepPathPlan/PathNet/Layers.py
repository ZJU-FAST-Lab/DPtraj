'''Define the Layers
Derived from - https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Layers.py
'''

import torch.nn as nn
import torch
import torch.utils.checkpoint
from SubLayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    ''' Single Encoder layer, that consists of a MHA layers and positiion-wise
    feedforward layer.
    '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v):
        '''
        Initialize the module.
        :param d_model: Dimension of input/output of this layer
        :param d_inner: Dimension of the hidden layer of hte position-wise feedforward layer
        :param n_head: Number of self-attention modules
        :param d_k: Dimension of each Key
        :param d_v: Dimension of each Value
        :param dropout: Argument to the dropout layer.
        '''
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner)

    def forward(self, enc_input, slf_attn_mask=None):
        '''
        The forward module:
        :param enc_input: The input to the encoder.
        :param slf_attn_mask: TODO ......
        '''
        # # Without gradient Checking
        # enc_output = self.slf_attn(
        #     enc_input, enc_input, enc_input, mask=slf_attn_mask)

        # With Gradient Checking
        # enc_output = torch.utils.checkpoint.checkpoint(self.slf_attn, 
        # enc_input, enc_input, enc_input, slf_attn_mask)
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, slf_attn_mask)

        # enc_output, enc_slf_attn = self.slf_attn(
        #     enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)
        return enc_output


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        '''
        Initialize the Layer
        :param d_model: Dimension of input/output this layer.
        :param d_inner: Dimension of hidden layer of the position wise FFN
        :param n_head: Number of self-attention modules.
        :param d_k: Dimension of each Key.
        :param d_v: Dimension of each Value.
        :param dropout: Argument to the dropout layer.
        '''
        super(DecoderLayer, self).__init__()
        # self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        '''
        Callback function
        :param dec_input:
        :param enc_output:
        :param slf_attn_mask:
        :param dec_enc_attn_mask:
        '''
        # dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_enc_attn