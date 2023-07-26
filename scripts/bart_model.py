import torch
from torch import nn

from transformers import BartForConditionalGeneration, BartTokenizer
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import transformers

class CopyLayerVocabExtend(torch.nn.Module):
  def __init__(self, num_embeddings):
    super().__init__()
    self.switch = torch.nn.Linear(num_embeddings, 1)

  def forward(self,
              src: torch.Tensor,
              output: torch.Tensor,
              attention: torch.Tensor):
    
    p_pointer = torch.sigmoid(self.switch(output))
    if torch.max(src) + 1 > output.shape[-1]:
      extended = Variable(torch.zeros((output.shape[0], output.shape[1], torch.max(src) + 1 - output.shape[-1]))).to(output.device)
      output = torch.cat((output, extended), dim = 2)
    generation_scores = (1 - p_pointer) * F.softmax(output, dim = 2)
    copy_scores = p_pointer * attention
    src_ids = src.unsqueeze(1).repeat(1, output.shape[1], 1)
    output = generation_scores.scatter_add(2, src_ids, copy_scores) + 1e-10
    
    return torch.log(output), attention
    
class BARTSeq2Seq(torch.nn.Module):
  def __init__(self, voc_size, unk_token, bart, copy_layer):
    super().__init__()
    self.voc_size = voc_size
    self.unk_token = unk_token
    self.bart = bart
    self.bart.resize_token_embeddings(voc_size)
    self.copy_layer = copy_layer
    if copy_layer:
      self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
      self.criterion = None
    self.optimizer=optim.AdamW(self.parameters(),lr=0.000015)
    self.lr_scheduler=transformers.get_polynomial_decay_schedule_with_warmup(self.optimizer, 5000, 30000,power=0.5)
    self.clip = None

  def forward(self, input_ids, trg_ids, mask, generation=False):
    decoder_input_ids = trg_ids#[:,:-1]
    labels = trg_ids#[:,1:]
    if generation:
        output_dict = self.bart(input_ids=input_ids.masked_fill(input_ids >= self.voc_size, self.unk_token), 
                                attention_mask=mask,
                                decoder_input_ids=decoder_input_ids.masked_fill(decoder_input_ids >= self.voc_size, self.unk_token),
                                output_hidden_states=True, 
                                return_dict=True, output_attentions=True)
    else:
        output_dict = self.bart(input_ids=input_ids.masked_fill(input_ids >= self.voc_size, self.unk_token), 
                                attention_mask=mask,
                                labels=labels.masked_fill(labels >= self.voc_size, self.unk_token),
                                output_hidden_states=True, 
                                return_dict=True, output_attentions=True)
    output = output_dict.logits
    attention = output_dict.cross_attentions[0]
    if self.copy_layer is not None:
        output, attention = self.copy_layer(input_ids, output, attention[:, -1, :, :])
        output_dim = output.shape[-1]
        out_reshaped = output.contiguous().view(-1, output_dim)
        trg_reshaped = trg_ids.contiguous().view(-1)
        # print(out_reshaped.shape, trg_reshaped.shape)
        # print(out_reshaped.max().item(), trg_reshaped.max().item())
        loss = self.criterion(out_reshaped, trg_reshaped)
    else:
        loss = output_dict.loss
        
    return output, attention, loss

  def generate(self, src, mask, max_length, use_copy, sos_token):
    self.eval()
    if use_copy:
      max_batch_len = max_length - 2
      prd = torch.full((src.shape[0], 1), sos_token).to(src.device)
      for test in range(max_length):
        out, attention, loss = self(src, prd, mask, True)
        pred_token = out.argmax(2)[:,-1]
        pred_token = torch.unsqueeze(pred_token, dim=1)
        prd = torch.cat((prd, pred_token), dim=1)
      return prd[:,1:]
    else:
      return self.bart.generate(input_ids=src, num_beams=1, attention_mask=mask, early_stopping=True, max_length=max_length,num_return_sequences=1)[:,1:]

  def beam_generate(self, src, mask, max_length, num_beams, num_returns):
    return self.bart.generate(input_ids=src, num_beams=num_beams, attention_mask=mask, early_stopping=True, max_length=max_length,num_return_sequences=num_returns)[:,1:]

  def save(self, folder, name, model_save_folder="~/scratch/results"):
    torch.save(self.state_dict(), f"{model_save_folder}/{folder}/{name}.pt")

  @staticmethod
  def create_config(src_dim, trg_dim, src_length, trg_length, src_padding_id, trg_padding_id, src_unk_token, trg_unk_token, max_length):
    return {
        "voc_size": src_dim,
        "unk_token": src_unk_token
    }

  @classmethod
  def load(cls, folder="", name="", config={}, device=None, use_copy=False, model_save_folder="~/scratch/results"):
      model = cls.create_model(config, use_copy, device)
      model.load_state_dict(torch.load(f'{model_save_folder}/{folder}/{name}.pt'))
      return model

  @staticmethod
  def create_model(config, use_copy, device):
    try: bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    except: bart = BartForConditionalGeneration.from_pretrained("bart")
    copy_layer = CopyLayerVocabExtend(config["voc_size"]) if use_copy else None
    return BARTSeq2Seq(config["voc_size"], config["unk_token"], bart, copy_layer)
