import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.nn.functional as F

from typing import List, Dict, Tuple, Optional

class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int, # SV (base source vocab length)
                 emb_dim: int,   # E (embed)
                 hid_dim: int,   # C (conv)
                 dropout: float,
                 device: torch.device,
                 padding_idx: int,
                 max_length: int):
        super().__init__()

        self.padding_idx = padding_idx

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, emb_dim, self.padding_idx)
        self.pos_embedding = nn.Embedding(max_length, emb_dim, self.padding_idx)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        in_channels = hid_dim

        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        # Hardcoded for our purposes, but definitely could be changed/passed as a parameter
        convolutions = [(hid_dim, 3, 1)] * 9 + [(2 * hid_dim, 3, 1)] * 4 + [(4 * hid_dim, 1, 1)] * 2
        layer_in_channels = [in_channels]
        
        for (out_channels, kernel_size, residual) in convolutions:
            
            residual_dim = layer_in_channels[-residual]
            self.projections.append(
                nn.Linear(residual_dim, out_channels)
                if residual_dim != out_channels
                else None
            )

            self.convolutions.append(
                nn.Conv1d(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    padding = kernel_size // 2
                )
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        
        self.hid2emb = nn.Linear(in_channels, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = src.shape[0] # B
        src_len = src.shape[1] # S (longest sentence in src batch)

        pos = torch.arange(src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)
        # pos = [0, 1, 2, 3, ..., src len - 1] -> B x S
        encoder_padding_mask = src.eq(self.padding_idx)  # B x S

        if not encoder_padding_mask.any():
          encoder_padding_mask = None
        else:
          pos.masked_fill_(encoder_padding_mask, self.padding_idx)

        # embed tokens and positions
        tok_embedded = self.tok_embedding(src) # B x S x E
        pos_embedded = self.pos_embedding(pos) # B x S x E

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded) # B x S x E

        # pass embedded through linear layer to convert from emb dim (E) to hid dim (C)
        x = self.emb2hid(embedded) # B x S x C

        # permute for convolutional layer
        x = x.permute(0, 2, 1) # B x C x S

        # convolutions
        residuals = [x]
        for proj, conv, res_layer in zip(
            self.projections, self.convolutions, self.residuals
        ):
            if res_layer > 0:
                residual = residuals[-res_layer]

                residual = residual
                if proj is not None:
                  residual = proj(residual.permute(0,2,1))
                  residual = residual.permute(0,2,1) # B x C x S
            else:
                residual = None

            if encoder_padding_mask is not None:
              x = x.masked_fill(encoder_padding_mask.unsqueeze(1), 0)

            x = conv(self.dropout(x)) # B x C x S
            x = F.glu(x, dim=1) # B x C x S

            if residual is not None:
              x = (x + residual) * math.sqrt(0.5)

            residuals.append(x)

        # permute and convert back to emb dim
        x = self.hid2emb(x.permute(0, 2, 1)) # B x S x E

        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
        
        # Element-wise sum output (conved) and input (embedded) to be used for attention
        combined = (x + embedded) * math.sqrt(0.5)

        # x -> B x S x E
        # combined -> B x S x E
        # encoder_paddding_mask -> # B x S

        return x, combined, encoder_padding_mask
    
class AttentionLayer(nn.Module):
    def __init__(self, conv_channels: int, embed_dim: int, bmm=None):
        super().__init__()
        self.in_projection = nn.Linear(conv_channels, embed_dim) # C to E
        self.out_projection = nn.Linear(embed_dim, conv_channels) # E to C
        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, 
                x: torch.Tensor, 
                target_embedding: torch.Tensor, 
                encoder_out: torch.Tensor, 
                encoder_padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x

        # x -> B x T x C
        # target_embedding -> # B x T x E
        # encoder_out[0] -> B x E x S
        # encoder_out[1] -> B x S x E
        # src_mask -> B x S

        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5) # B x T x E
        x = self.bmm(x, encoder_out[0]) # B x T x E

        if encoder_padding_mask is not None:
            x = (
                x.float()
                .masked_fill(encoder_padding_mask.unsqueeze(1), float("-inf"))
                .type_as(x)
            ) # B x T x S

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x # B x T x S

        x = self.bmm(x, encoder_out[1]) # B x T x E

        # scale attention output (respecting potentially different lengths)
        s = encoder_out[1].size(1) # S

        if encoder_padding_mask is None:
            x = x * (s * math.sqrt(1.0 / s)) # B x T x E
        else:
            s = s - encoder_padding_mask.type_as(x).sum(
                dim=1, keepdim=True
            )  # exclude padding
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt()) # B x T x E

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5) # B x T x C

        # attn_scores -> T x S
        return x, attn_scores
    
class Decoder(nn.Module):
    def __init__(self,
                 num_embeddings: int, # TV (base target vocab length)
                 output_dim: int,     # O 
                 emb_dim: int,        # E
                 hid_dim: int,
                 dropout: float,
                 device: torch.device,
                 padding_idx: int,
                 max_length: int):
        super().__init__()

        self.padding_idx = padding_idx

        self.device = device

        self.tok_embedding = nn.Embedding(num_embeddings, emb_dim, self.padding_idx)
        self.pos_embedding = nn.Embedding(max_length, emb_dim, self.padding_idx)

        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.residuals = []

        self.dropout = nn.Dropout(dropout)

        in_channels = hid_dim
        layer_in_channels = [in_channels]

        # Should be the same as in the encoder
        convolutions = [(hid_dim, 3, 1)] * 9 + [(2*hid_dim, 3, 1)] * 4 + [(4*hid_dim, 1, 1)] * 2

        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            residual_dim = layer_in_channels[-residual] # hid dim
            self.projections.append(
                nn.Linear(residual_dim, out_channels) 
                if residual_dim != out_channels
                else None
            )

            self.convolutions.append(
                nn.Conv1d(
                    in_channels,
                    out_channels * 2,
                    kernel_size
                )
            )           

            self.attention.append(
                AttentionLayer(out_channels, emb_dim)
            )

            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(in_channels, output_dim)
        self.fc_out = nn.Linear(output_dim, num_embeddings)

    def forward(self, 
                trg: torch.Tensor, 
                encoder_conved: torch.Tensor, 
                encoder_combined: torch.Tensor, 
                src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # encoder_conved -> B x S x E
        # encoder_combined -> B x S x E
        # src_mask -> # B x S
        # trg -> # B x T

        batch_size = trg.shape[0] # B
        trg_len = trg.shape[1] # T (longest sentence in targe batch)

        encoder_conved = encoder_conved.permute(0, 2, 1) # B x E x S

        # create position tensor
        pos = torch.arange(trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        # pos = B x T

        decoder_padding_mask = trg.eq(self.padding_idx)  # -> B x T
        if not decoder_padding_mask.any():
          decoder_padding_mask = None
        else:
          pos.masked_fill_(decoder_padding_mask, self.padding_idx)
        
        # embed tokens and positions
        tok_embedded = self.tok_embedding(trg) # B x T x E
        pos_embedded = self.pos_embedding(pos) # B x T x E

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded) # B x T x E

        # pass embedded through linear layer to go through emb dim (E) -> hid dim (C)
        x = self.emb2hid(embedded) # B x T x C

        batch_size = x.shape[0]
        residuals = [x]
        for proj, conv, attention, res_layer in zip(
            self.projections, self.convolutions, self.attention, self.residuals
        ):
            if res_layer > 0:
                residual = residuals[-res_layer] # B x T x C
                residual = residual if proj is None else proj(residual)  # B x T x C
            else:
                residual = None

            hid_dim = x.shape[2] # C
            x = self.dropout(x) # B x T x C
            # we permute it here as opposed to beforer the conv in the encoder because of the attn layer
            x = x.permute(0, 2, 1) # B x C x T

            # K is kernel size
            padding = torch.zeros(batch_size, hid_dim, conv.kernel_size[0] - 1) # B x C x K
            padding = padding.fill_(self.padding_idx).to(self.device) # B x C x K
            padded_x = torch.cat((padding, x), dim = 2) # B x C x [K + T]

            x = conv(padded_x)     # B x C x [K + T]
            x = F.glu(x, dim=1)    # B x C x [K + T]
            x = x.permute(0, 2, 1) # B x [K + T] x C

            if attention is not None:
                attn, attn_scores = attention(
                    x, embedded, (encoder_conved, encoder_combined), src_mask

                )
                
            if residual is not None:
              x = (attn + residual) * math.sqrt(0.5) # B x T x C

            residuals.append(x)

        x = self.hid2emb(x) # B x T x E
        output = self.fc_out(self.dropout(x)) # B x T x O
        # attn_scores -> S x T
        return output, attn_scores

class CopyLayerVocabExtend(nn.Module):
  def __init__(self, decoder: Decoder):
    super().__init__()
    self.switch = nn.Linear(decoder.tok_embedding.num_embeddings, 1)

  def forward(self,
              src: torch.Tensor, 
              output: torch.Tensor, 
              attention: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    p_pointer = torch.sigmoid(self.switch(output))
  
    if torch.max(src) + 1 > output.shape[-1]:
      extended = Variable(torch.zeros((output.shape[0], output.shape[1], torch.max(src) + 1 - output.shape[-1]))).to(output.device)
      output = torch.cat((output, extended), dim = 2)

    output = ((1 - p_pointer) * F.softmax(output, dim = 2)).scatter_add(2, src.unsqueeze(1).repeat(1, output.shape[1], 1), p_pointer * attention) + 1e-10
    return torch.log(output), attention

class CNNSeq2Seq(nn.Module):
    
    encoder_params = ["input_dim", "emb_dim", "hid_dim", "dropout", "device", "padding_idx", "max_length"]
    decoder_params = ["num_embeddings", "output_dim", "emb_dim", "hid_dim", "dropout", "device", "padding_idx", "max_length"]

    HP = {
        "optimizer": {
            "lr": 0.5,
            "momentum": 0.9,
        },
        "clip": 0.1,
        "criterion": {
            "label_smoothing": 0.1
        },
        "encoder": {
            "emb_dim": 768, # size of the embeddings
            "hid_dim": 512, # encoder hidden state
            "dropout": 0.2  # encoder dropout
        },
        "decoder":{
            "emb_dim" : 768, 
            "output_dim": 512,
            "hid_dim" : 512, 
            "dropout" : 0.2,
        }

    }
    
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 src_unk_token: int,
                 trg_unk_token: int,
                 trg_pad_token: int,
                 copy_layer: Optional[CopyLayerVocabExtend]=None):
        super().__init__()
        self.src_unk_token = src_unk_token
        self.trg_unk_token = trg_unk_token
        self.encoder = encoder
        self.decoder = decoder
        self.copy_layer = copy_layer
        self.optimizer = torch.optim.SGD(self.parameters(), **self.HP["optimizer"])
        self.lr_scheduler = None 
        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_token, **self.HP["criterion"])
        self.clip = self.HP["clip"]

    def forward(self, src, trg, mask) -> Tuple[torch.Tensor, torch.Tensor]:
      trg_input = trg[:,:-1]
      if self.copy_layer is None:
        encoder_conved, encoder_combined, encoder_padding_mask = self.encoder(src)
        output, attention = self.decoder(trg_input, encoder_conved, encoder_combined, encoder_padding_mask)

      else:
        encoder_conved, encoder_combined, encoder_padding_mask = self.encoder(src.masked_fill(src >= self.encoder.tok_embedding.num_embeddings, 0))
        output, attention = self.decoder(trg_input.masked_fill(trg_input >= self.decoder.tok_embedding.num_embeddings, 0), encoder_conved, encoder_combined, encoder_padding_mask)
        output, attention = self.copy_layer(src, output, attention)

      output_dim = output.shape[-1]
      out_reshape = output.contiguous().view(-1, output_dim)
      trg_label = trg[:,1:].contiguous().view(-1)
      loss = self.criterion(out_reshape, trg_label)
      return output, attention, loss
          
    def generate(self, src, mask, max_length, use_copy, sos_token):
      self.eval()
      max_batch_len = max_length - 2
        
      with torch.no_grad():
        encoder_conved, encoder_combined, encoder_padding_mask = self.encoder(src.masked_fill(src >= self.encoder.tok_embedding.num_embeddings, 0))
    
      # init empty query sentence with only <sos> token
      prd = torch.full((src.shape[0], 1), sos_token).to(src.device)
    
      # generate words
      # for test in tqdm(range(max_length)):
      for test in range(max_batch_len):
        with torch.no_grad():
            out, att = self.decoder(prd.masked_fill(prd >= self.decoder.tok_embedding.num_embeddings, 0), encoder_conved, encoder_combined, encoder_padding_mask)
            if use_copy:
                out, att = self.copy_layer(src, out, att)
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
        
    @classmethod
    def create_config(cls, src_dim, trg_dim, src_length, trg_length, src_padding_id, trg_padding_id, max_length, src_unk_token, trg_unk_token):
        config = {}

        # Adding encoder params
        config["encoder"] = cls.HP["encoder"].copy()
        config["encoder"].update({
            "input_dim": src_dim,
            "padding_idx": src_padding_id, 
            "max_length": max_length 
        })
        # Adding decoder params
        config["decoder"] = cls.HP["decoder"].copy()
        config["decoder"].update({
            "num_embeddings": trg_dim, # trg vocab size
            # "output_dim": trg_dim, # decoder output size
            "padding_idx": trg_padding_id, 
            "max_length": max_length
        })
        # Adding Seq2Seq params
        config.update({
            "src_unk_token": src_unk_token,
            "trg_unk_token": trg_unk_token,
        })
        return config
        
    @staticmethod
    def create_model(config, use_copy, device):
        # print(config)
        encoder_config = config["encoder"].copy()
        encoder_config.update({"device": device})
        decoder_config = config["decoder"].copy()
        decoder_config.update({"device": device})
        encoder = Encoder(**encoder_config)
        decoder = Decoder(**decoder_config)
        copy_layer = CopyLayerVocabExtend(decoder) if use_copy else None
        return CNNSeq2Seq(encoder, decoder, config["src_unk_token"], config["trg_unk_token"], config["decoder"]["padding_idx"], copy_layer)
