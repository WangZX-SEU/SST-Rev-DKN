import copy

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import argparse
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from copy import deepcopy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, data_tensor1, data_tensor2):
        self.data_tensor1 = data_tensor1
        self.data_tensor2 = data_tensor2

    def __len__(self):
        return self.data_tensor1.shape[1]

    def __getitem__(self, idx):
        input_sequence = self.data_tensor1[:, idx, :]
        target_sequence= self.data_tensor2[:, idx, :]

        return input_sequence, target_sequence

class MyDataset2(Dataset):
    def __init__(self, data_tensor1):
        self.data_tensor1 = data_tensor1

    def __len__(self):
        return self.data_tensor1.shape[1]

    def __getitem__(self, idx):
        input_sequence = self.data_tensor1[:, idx, :]

        return input_sequence

class SelfSupervisedTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,       # encoder layer
                 num_decoder_layers: int,       # decoder layer
                 emb_size: int,                 # embedding dimension
                 nhead: int,                    # multi-head
                 state_size: int,               # source size
                 tgt_size: int,                 # target size
                 batch_size: int,               # batch_size
                 memory_step: int,              # memory window length
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(SelfSupervisedTransformer, self).__init__()

        self.SRC_SIZE = state_size
        self.TGT_IZE = tgt_size
        self.X_EMB_SIZE = emb_size
        self.NHEAD = nhead
        self.FFN_HID_DIM = dim_feedforward
        self.NUM_ENCODER_LAYERS = num_encoder_layers
        self.NUM_DECODER_LAYERS = num_decoder_layers
        self.MEMORY_STEP = memory_step
        self.BATCH_SIZE = batch_size

        self.transformers = Transformer(d_model=emb_size,
                                        nhead=nhead,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout)

        self.Pool = nn.AdaptiveAvgPool3d(output_size=(1, self.BATCH_SIZE, emb_size))
        self.generator = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(self.SRC_SIZE, self.X_EMB_SIZE)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):

        src_emb = self.PositionalEncoding(src)
        tgt_emb = self.PositionalEncoding(trg)
        outs = self.transformers(src_emb, tgt_emb, src_mask, tgt_mask)

        self.Pool.output_size = (1, min(self.BATCH_SIZE, src_emb.shape[1]), self.X_EMB_SIZE)

        return self.dropout(self.generator(self.Pool(outs.unsqueeze(0)).squeeze(0)))

    def generate_square_subsequent_mask(self, sz):
        '''
            [0 0 0 -inf -inf -inf]
            [0 0 0   0  -inf -inf]
            [0 0 0   0    0  -inf]
            [0 0 0   0    0    0 ]
        '''
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0,
                                                                                1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)  # 目标词语mask操作，防止self-attention关注未来信息
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def Embedding(self, tokens):
        tokens = tokens.to(torch.float64)
        return self.embedding(tokens) * math.sqrt(self.X_EMB_SIZE)

    def PositionalEncoding(self, x, maxlen: int = 5000):
            den = torch.exp(- torch.arange(0, self.X_EMB_SIZE, 2) * math.log(10000) / self.X_EMB_SIZE)
            pos = torch.arange(0, maxlen).reshape(maxlen, 1)
            pos_embedding = torch.zeros((maxlen, self.X_EMB_SIZE))

            pos_embedding[:, 0::2] = torch.sin(pos * den)
            pos_embedding[:, 1::2] = torch.cos(pos * den)

            pos_embedding = pos_embedding.unsqueeze(1)

            return self.dropout(x + pos_embedding[:x.size(0), :, :].detach())

    def encoder(self, src, last_epoch_x, first_epoch=False):
        src = src.permute(1, 0, 2).to(torch.float32).to(DEVICE)
        src = self.Embedding(src)
        length = src.shape[0]

        out_data = np.zeros((src.shape[0], src.shape[1], self.X_EMB_SIZE))

        for iter_count in range(length):
            if iter_count <= self.MEMORY_STEP - 1:
                src_memory = src[:iter_count + 1, :, :]
                target_shape = (self.MEMORY_STEP, self.BATCH_SIZE, self.X_EMB_SIZE)
                pad = [(target_shape[i] - src_memory.shape[i]) for i in range(len(target_shape))]
                temp =src_memory[-1, :, :].repeat(pad[0], 1, 1)
                src_memory = torch.cat((src_memory, temp), dim=0)
                if first_epoch:
                    tgt_memory = deepcopy(src_memory.detach())
                    tgt_memory.requires_grad = True
                else:
                    tgt_memory = last_epoch_x[:iter_count + 1, :, :].to(torch.float64)
            else:
                src_memory = src[iter_count - self.MEMORY_STEP + 1:iter_count + 1, :, :]
                if first_epoch:
                    tgt_memory = deepcopy(src_memory.detach())
                    tgt_memory.requires_grad = True
                else:
                    tgt_memory = last_epoch_x[iter_count - self.MEMORY_STEP + 1:iter_count + 1, :, :].to(torch.float64)

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src_memory, tgt_memory)

            out = self.forward(src_memory, tgt_memory, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,
                               src_padding_mask)

            out_data[iter_count, :, :] = out.detach().numpy()

        return torch.tensor(out_data, requires_grad=True)

    def encode(self, src: Tensor, src_mask: Tensor):
        src = self.Embedding(src)
        return self.transformers.encoder(self.PositionalEncoding(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt = tgt.to(torch.float64)
        return self.transformers.decoder(self.PositionalEncoding(tgt), memory, tgt_mask)


