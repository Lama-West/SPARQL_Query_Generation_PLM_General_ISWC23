import torch
from torch import nn
from torchtext.data.metrics import bleu_score
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
# import wandb

try:
  from data import *
  from consts import *
  from t5_model import *
  from bart_model import *
except:
  from scripts.data import *
  from scripts.consts import *
  from scripts.t5_model import *
  from scripts.bart_model import *


def initialize_weights(m) -> None:
    if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def remove_special(batch_ids, vocab):
  # for i in range(len(batch_ids)):
  #   batch_
  # for ids in batch_ids:
  #   if "<sos>" in ids:
  #     ids.remove("<sos>")
  return [[token for token in ids if token not in vocab.special_tokens] for ids in batch_ids]

def compute_bleu(trg, out, vocab):
  trg_ids = vocab.get_tokens(trg.cpu().numpy(), "trg")
  out_ids = vocab.get_tokens(out.argmax(2).cpu().numpy(), "trg")
  # print(trg_ids, out_ids)
  trg_ids = remove_special(trg_ids, vocab)
  out_ids = remove_special(out_ids, vocab)
  return bleu_score(out_ids, [[ids] for ids in trg_ids])

def show_example(src, trg, out, vocab, i, beams=None):
  id = np.random.randint(trg.shape[0])
  trg_ids = vocab.get_tokens(trg.cpu().numpy(), "trg")
  out_ids = vocab.get_tokens(out.argmax(2).cpu().numpy(), "trg")
  src_ids = vocab.get_tokens(src.cpu().numpy(), "src")
  print("     Epoch: ->", i, flush=True)
  print("    Source: ->", " ".join(src_ids[id]), flush=True)
  src_ids = remove_special(src_ids, vocab)
  # print(src_ids[id])
  print("    Target: ->", " ".join(trg_ids[id]), flush=True)
  trg_ids = remove_special(trg_ids, vocab)
  # print(trg_ids[id])
  print("Prediction: ->", " ".join(out_ids[id]), flush=True)
  out_ids = remove_special(out_ids, vocab)
  # print(out_ids[id])
  if beams is not None:
    for b, beam in enumerate(beams[id]):
      print(f"  Beam {b}: ->", beam, flush=True)

def save_training_config(folder, epoch, batch, state):
  with open(f"{model_save_folder}/{folder}/config.json") as file:
      config = json.load(file)
  config["training"].update( {
      "current_epoch": epoch,
      "current_batch": batch,
      "state": state
  } )
  with open(f"{model_save_folder}/{folder}/config.json", "w") as file:
      file.write(json.dumps(config))

def get_training_metrics(folder):
  with open(f"{model_save_folder}/{folder}/training_metrics.json") as file:
      training_metrics = json.load(file)
  return training_metrics

def save_training_metrics(folder, training_metrics):
  with open(f"{model_save_folder}/{folder}/training_metrics.json", "w") as file:
    file.write(json.dumps(training_metrics))

def plot_losses(training_metrics, folder, artist):
  fig, hfig, ax1, ax2, ax3, ax4 = artist
  for ax in (ax1, ax2, ax3, ax4):
    ax.cla()
  lines = []
  lines += ax1.plot(training_metrics["train_loss"], label="train loss", c="cyan", linestyle="--")
  lines += ax1.plot(training_metrics["val_loss"], label="val loss", c="b", linestyle="--")
  best_loss_id = np.argmin(training_metrics["val_loss"])
  ax1.scatter([best_loss_id], [training_metrics["val_loss"][best_loss_id]], c="r")
  lines += ax2.plot(training_metrics["train_bleu"], label="train bleu", c="orange") 
  lines += ax2.plot(training_metrics["val_bleu"], label="val bleu", c="g") 
  best_bleu_id = np.argmax(training_metrics["val_bleu"])
  ax2.scatter([best_bleu_id], [training_metrics["val_bleu"][best_bleu_id]], c="r")

  ax2.legend(lines, [l.get_label() for l in lines], loc=0)

  lines = []
  lines += ax3.plot(training_metrics["train_loss"][-10:], label="train loss", c="cyan", linestyle="--")
  lines += ax3.plot(training_metrics["val_loss"][-10:], label="val loss", c="b", linestyle="--")
  best_loss_id = np.argmin(training_metrics["val_loss"][-10:])
  if np.min(training_metrics["val_loss"][-10:]) == np.min(training_metrics["val_loss"]):
    color = "g"
  else:
    color = "r"
  ax3.scatter([best_loss_id], [training_metrics["val_loss"][-10:][best_loss_id]], c=color)
  lines += ax4.plot(training_metrics["train_bleu"][-10:], label="train bleu", c="orange") 
  lines += ax4.plot(training_metrics["val_bleu"][-10:], label="val bleu", c="g") 
  best_bleu_id = np.argmax(training_metrics["val_bleu"][-10:])
  if np.max(training_metrics["val_bleu"][-10:]) == np.max(training_metrics["val_bleu"]):
    color = "g"
  else:
    color = "r"
  ax4.scatter([best_bleu_id], [training_metrics["val_bleu"][-10:][best_bleu_id]], c=color)

  ax4.legend(lines, [l.get_label() for l in lines], loc=0)

  plt.savefig(f"images/{folder}.png")
  plt.savefig(f"{model_save_folder}/{folder}/training.png")
  # wandb.log({"val loss": training_metrics["val_loss"][-1], 
  #            "train_loss": training_metrics["train_loss"][-1],
  #            "val_bleu": training_metrics["val_bleu"][-1],
  #            "train_bleu": training_metrics["train_bleu"][-1],
  #            })
  if hfig is not None:
    fig.canvas.draw()
    hfig.update(fig)

def initialize_artist():
  fig, (ax1, ax3) = plt.subplots(1,2)
  ax2 = ax1.twinx()
  ax4 = ax3.twinx()
  fig.set_size_inches(18.5, 7)
  try: 
    disp = display(fig, display_id=True)
  except:
    disp = None
  return fig, disp, ax1, ax2, ax3, ax4

def save_partial_report(report, gcn):
  if os.path.isfile(f"reports/{gcn}/report_temp.json"):
    with open(f"reports/{gcn}/report_temp.json") as file:
      previous_reports = json.load(file)
    report += previous_reports
  with open(f"reports/{gcn}/report_temp.json", "w") as file:
      file.write(json.dumps(report))

def train(model, train_dataloader, val_dataloader, device, vocab, train_config, folder):
  model = model.to(device)

  training_metrics = get_training_metrics(folder)
  min_loss = 1e9

  artist = initialize_artist()

  for epoch in tqdm(range(train_config["epochs"]), desc='Training'):
  #for epoch in range(train_config["epochs"]):
    # print(f"Epoch {epoch + 1}/{train_config['epochs']}")
    if epoch < train_config["current_epoch"]: continue
    t = time.time()
    
    epoch_loss = []
    epoch_bleu = []
    # for batch_i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):#, position=0, leave=True)):
    for batch_i, batch in tqdm(list(enumerate(train_dataloader)), desc=f"Epoch {epoch + 1}") if "dbnqa" in folder else enumerate(train_dataloader):
      model.train()
      
      if epoch == train_config['current_epoch'] and batch_i < train_config["current_batch"]: 
        continue

      model.optimizer.zero_grad()
      src, mask, trg, ids = batch
      src = src.to(device)
      trg = trg.to(device)
      mask = mask.to(device)
      out, attention, loss = model(src, trg, mask)
      
      loss.backward()
      if model.clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.clip)
      model.optimizer.step()
      if model.lr_scheduler is not None: 
        model.lr_scheduler.step()

      epoch_loss += [loss.item()]
      epoch_bleu += [compute_bleu(trg, out, vocab)]

      if time.time() - t > 5 * 60:
        save_training_config(folder, epoch, batch_i, "running")
        save_training_metrics(folder, training_metrics)
        model.save(folder, "current-model", model_save_folder=model_save_folder)
        t = time.time()
    # print("End training epoch") 
    val_loss, val_bleu = evaluate(model, val_dataloader, device, vocab, epoch + 1, folder, train_config["epochs"])

    training_metrics["train_loss"] += [np.mean(epoch_loss)]
    training_metrics["val_loss"] += [val_loss]
    training_metrics["train_bleu"] += [np.mean(epoch_bleu)]
    training_metrics["val_bleu"] += [val_bleu]
    
    if val_loss < min_loss:
      # print("val_loss", val_loss)
      # print("min_loss", min_loss)
      # print("val_losses", training_metrics["val_loss"])
      min_loss = val_loss
      model.save(folder, "best-model", model_save_folder = model_save_folder)

    save_training_config(folder, epoch + 1, 0, "running")
    save_training_metrics(folder, training_metrics)
    model.save(folder, "current-model", model_save_folder=model_save_folder)
    plot_losses(training_metrics, folder, artist)

  save_training_config(folder, epoch, batch_i, "finished")
  save_training_metrics(folder, training_metrics)

def evaluate(model, val_dataloader, device, vocab, epoch, folder, epochs):
  model.eval()
  epoch_loss = []
  epoch_bleu = []

  with torch.no_grad():

    for _, batch in enumerate(val_dataloader):

      src, mask, trg, ids = batch
      src = src.to(device)
      trg = trg.to(device)
      mask = mask.to(device)

      out, attention, loss = model(src, trg, mask)

      epoch_loss += [loss.item()]

      bleu_batch = compute_bleu(trg, out, vocab)
      epoch_bleu.append(bleu_batch)

    if hasattr(model, "beam_generate") and model.copy_layer is None:
        out_beams = model.beam_generate(src, mask, max_length=150, num_beams=5, num_returns=5)
        out_sentences = [sentence.split(vocab.tokenizer.eos_token)[0].replace("<s>", "") for sentence in vocab.tokenizer.batch_decode(out_beams[:,1:])]
        out_sentences = np.array(out_sentences).reshape(src.shape[0], 5)
    else:
        out_sentences = None
  
  # print("Evaluating epoch")
  if epoch % ((epochs // 50)+1)  == 0:
    # print("Show batch")
    show_example(src, trg, out, vocab, epoch, out_sentences)
  return np.mean(epoch_loss), np.mean(epoch_bleu)

def test(model, test_dataloader, device, vocab, max_length, use_copy, folder, num_beams=10, num_beams_return=10):
  model.eval()

  report = []
  t = time.time()
  for i, batch in enumerate(tqdm(list(test_dataloader))):

    src, mask, trg, ids = batch
    src = src.to(device)
    trg = trg.to(device)
    mask = mask.to(device)
    if isinstance(model, BARTSeq2Seq):
      sos_token_id = vocab.tokenizer.bos_token_id
    elif isinstance(model, T5Seq2Seq):
      sos_token_id = model.t5.config.pad_token_id # same as bos
    else:
      sos_token_id = vocab.vocab["trg"]["stoi"][vocab.sos_token]
    with torch.no_grad():
      prd_ids = model.generate(src, mask, max_length, use_copy, sos_token_id)

    prd_tokens = vocab.get_tokens(prd_ids.cpu().numpy(), "trg")

    if hasattr(model, "beam_generate") and model.copy_layer is None:
      out_beams = model.beam_generate(src, mask, max_length=max_length, num_beams=num_beams, num_returns=num_beams_return)
      all_beams_ids = out_beams.detach().cpu().reshape(src.shape[0], num_beams_return, out_beams.shape[1])
      all_out_sentences = []
      for batch_beam_ids in all_beams_ids:
        batch_beams_tokens = vocab.get_tokens(batch_beam_ids, None)
        all_out_sentences.append([sentence[1:] for sentence in batch_beams_tokens])
    else:
      all_out_sentences = np.zeros((src.shape[0], num_beams_return)).tolist()

    trg_tokens = vocab.get_tokens(trg.cpu().numpy()[:,1:] if "t5" not in folder else trg.cpu().numpy(), "trg")
    src_tokens = vocab.get_tokens(src.cpu().numpy()[:,1:], "src")
    ids = ids.numpy()

    for src, trg, prd, batch_out_sentences, id in zip(src_tokens, trg_tokens, prd_tokens, all_out_sentences, ids):
      report.append({
          "src": src,
          "trg": trg,
          "prd": prd[1:] if "t5" not in folder else prd,
          "prd_topn_beams": batch_out_sentences,
          "id": id.item(),
          "correct": trg == (prd[1:] if "t5" not in folder else prd)
      })

    # if time.time() - t > 60 * 5:
    #   t = time.time()
    #   print(len(report))
    #   save_partial_report(report, folder)

  return report
