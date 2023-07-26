import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch import tensor
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math
import numpy as np
from tqdm import tqdm

from typing import List, Dict, Tuple


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, 
                 hid_dim:int, 
                 n_heads:int, 
                 dropout:float, 
                 device: torch.device):

        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x, attention
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, 
                 hid_dim: int, 
                 pf_dim: int, 
                 dropout: float):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, 
                 hid_dim: int, 
                 pf_dim: int, 
                 dropout: float):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim: int,
                 n_heads: int,
                 pf_dim: int,
                 dropout: float,
                 device: torch.device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device) 
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                src: torch.Tensor,
                src_mask: torch.Tensor) -> torch.Tensor:

        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src
    
class Encoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hid_dim: int, 
                 n_layers: int,
                 n_heads: int, 
                 pf_dim: int, 
                 dropout: float, 
                 device: torch.device, 
                 max_length: int):
        super().__init__()

        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, 
                src: torch.Tensor, 
                src_mask: torch.Tensor) -> torch.Tensor:
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(
            0).repeat(batch_size, 1).to(self.device)
        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            src = layer(src, src_mask)

        return src
    
class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim: int,
                 n_heads: int,
                 pf_dim: int,
                 dropout: float,
                 device: torch.device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                trg: torch.Tensor, 
                enc_src: torch.Tensor, 
                trg_mask: torch.Tensor, 
                src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # encoder attention
        _trg, attention = self.encoder_attention(
            trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention

class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 hid_dim: int,
                 n_layers: int,
                 n_heads: int,
                 pf_dim: int,
                 dropout: float,
                 device: torch.device,
                 max_length: int):
        super().__init__()

        self.device = device
        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout_val = dropout

        self.dropout = nn.Dropout(self.dropout_val)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, 
                trg: torch.Tensor, 
                enc_src: torch.Tensor, 
                trg_mask: torch.Tensor, 
                src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(
            0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention
    
class CopyLayerVocabExtend(nn.Module):
  def __init__(self, decoder: Decoder):
    super().__init__()
    self.switch = nn.Linear(decoder.tok_embedding.num_embeddings, 1)

  def forward(self, 
              src: torch.Tensor, 
              output: torch.Tensor, 
              attention: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                  
                  

    
    
    p_pointer = torch.sigmoid(self.switch(output))

    # src -> [2, 23, 1129, 40, 1083, 11, 3]
    # output = trg_predit -> [2, 45, 35, 1129, 40, 1083, 12, 3]
    
    if torch.max(src) + 1 > output.shape[-1]: # mots inconnus dans source?
      extended = Variable(torch.zeros((output.shape[0], output.shape[1], torch.max(src) + 1 - output.shape[-1]))).to(output.device)
      output = torch.cat((output, extended), dim = 2)

    output = ((1 - p_pointer) * F.softmax(output, dim = 2)).scatter_add(2, src.unsqueeze(1).repeat(1, output.shape[1], 1), p_pointer * attention[:, 3]) + 1e-10
    # output = trg_predit -> [2, 45, 35, 1129, 40, 1083, 12, 3]
    return torch.log(output), attention

class TransfSeq2Seq(nn.Module):
    
    encoder_params = ["input_dim", "hid_dim", "n_layers", "n_heads", "pf_dim", "dropout", "device", "max_length"]
    decoder_params = ["output_dim", "hid_dim", "n_layers", "n_heads", "pf_dim", "dropout", "device", "max_length"]
    
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_pad_idx: int,
                 trg_pad_idx: int,
                 src_unk_idx: int,
                 trg_unk_idx: int,
                 device: torch.device,
                 copy_layer: CopyLayerVocabExtend =None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_unk_idx = src_unk_idx
        self.trg_unk_idx = trg_unk_idx
        self.device = device
        self.copy_layer = copy_layer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005, weight_decay=0.0001, eps=1e-09, betas=(0.9,0.98))
        self.lr_scheduler = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)
        self.clip = 1

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=self.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg, mask):
        trg_input = trg[:,:-1]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg_input)

        if self.copy_layer is None:
          source = self.encoder(src, src_mask)
          output, attention = self.decoder(trg_input, source, trg_mask, src_mask)
        #   return output, attention, None

        else: # avec copie
          # src -> [2, 23, 1129, 40, 1083, 11, 3]
          # trg -> [2, 45, 34, 1129, 40, 1083, 12, 3]

          # src -> [2, 23, 0, 40, 0, 11, 3]
          source = self.encoder(src.masked_fill(src >= self.encoder.tok_embedding.num_embeddings, self.src_unk_idx), src_mask)

          # trg -> [2, 45, 34, 0, 40, 0, 12, 3]
          output, attention = self.decoder(trg_input.masked_fill(trg_input >= self.decoder.tok_embedding.num_embeddings, self.trg_unk_idx), source, trg_mask, src_mask)


          # src -> [2, 23, 1129, 40, 1083, 11, 3]
          # output = trg_predit -> [2, 45, 35, 0, 40, 0, 12, 3]
          output, attention = self.copy_layer(src, output, attention)
        #   return output, attention, None

        output_dim = output.shape[-1]
        out_reshape = output.contiguous().view(-1, output_dim)
        trg_reshaped = trg[:,1:].contiguous().view(-1)
        loss = self.criterion(out_reshape, trg_reshaped)
        return output, attention, loss
        
    def generate(self, src, mask, max_length, use_copy, sos_token):
      self.eval()
      max_batch_len = max_length - 2
      src_mask = self.make_src_mask(src)
      with torch.no_grad():
        enc_src = self.encoder(src.masked_fill(src >= self.encoder.tok_embedding.num_embeddings, self.src_unk_idx), src_mask)
    
    
      # init empty query sentence with only <sos> token
      prd = torch.full((src.shape[0], 1), sos_token).to(src.device)
    
      # generate words
      #for test in tqdm(range(max_length)):
      for test in range(max_length):
        trg_mask = self.make_trg_mask(prd)
        with torch.no_grad():
            out, att = self.decoder(prd.masked_fill(prd >= self.decoder.tok_embedding.num_embeddings, self.trg_unk_idx), enc_src, trg_mask, src_mask)
            if use_copy:
                out, att = self.copy_layer(src, out, att)
        # out = out[:, -1, :]
        pred_token = out.argmax(2)[:,-1]
        pred_token = torch.unsqueeze(pred_token, dim=1)
        prd = torch.cat((prd, pred_token), dim=1)
    
      return prd
       
    def save(self, folder, name, model_save_folder="~/scratch/results"):
        torch.save(self.state_dict(), f"{model_save_folder}/{folder}/{name}.pt")
        
    @classmethod
    def load(cls, folder="", name="", config={}, device=None, use_copy=False, model_save_folder="~/scratch/results"):
        model = cls.create_model(config, use_copy, device)
        model.load_state_dict(torch.load(f'{model_save_folder}/{folder}/{name}.pt', map_location=device))
        return model
        
    @staticmethod
    def create_config(src_dim, trg_dim, src_length, trg_length, src_padding_id, trg_padding_id, max_length, 
    src_unk_token=-1, 
    trg_unk_token=-1,
    hid_dim=1024, 
    n_layers=6, 
    n_heads=4,
    pf_dim=512,
    dropout=0.3
    ):
        
        config = {
            "pad_idx":{
                "src_padding_id": src_padding_id,
                "trg_padding_id": trg_padding_id
            },
            "src_unk_token": src_unk_token,
            "trg_unk_token": trg_unk_token,
            "encoder":{
                "input_dim": src_dim,
                "hid_dim":hid_dim, 
                "n_layers":n_layers, 
                "n_heads":n_heads, 
                "pf_dim":pf_dim, 
                "dropout":dropout,
                "max_length":max_length
            },
            "decoder":{
                "output_dim":trg_dim, 
                "hid_dim":hid_dim, 
                "n_layers":n_layers, 
                "n_heads":n_heads, 
                "pf_dim":pf_dim, 
                "dropout":dropout, 
                "max_length":max_length
                }
            }
        return config
        
    @staticmethod
    def create_model(config, use_copy, device):
        encoder_config = config["encoder"].copy()
        decoder_config = config["decoder"].copy()
        
        encoder_config["device"] = device
        decoder_config["device"] = device
        
        encoder = Encoder(**encoder_config)
        decoder = Decoder(**decoder_config)
        
        copy_layer = CopyLayerVocabExtend(decoder) if use_copy else None
        
        src_pad_id = config["pad_idx"]["src_padding_id"]
        trg_pad_id = config["pad_idx"]["trg_padding_id"]
        src_unk_token = config["src_unk_token"]
        trg_unk_token = config["trg_unk_token"]
        return TransfSeq2Seq(encoder, decoder, src_pad_id, trg_pad_id, src_unk_token, trg_unk_token, device, copy_layer)

