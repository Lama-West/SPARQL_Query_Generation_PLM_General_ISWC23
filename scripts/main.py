import os
import json
import sys
import shutil

try: 
  from consts import *
  from data import *
  from train import *
except:
  from scripts.consts import *
  from scripts.data import *
  from scripts.train import *

print("Importing main")

def build_vocab_config(vocab, use_copy, max_length, preprocessing_config):
  src_dim = vocab.get_dim("src", not use_copy)
  trg_dim = vocab.get_dim("trg", not use_copy)
  return {
      "src_dim": src_dim,
      "trg_dim": trg_dim,
      "src_length": preprocessing_config["src_padding"],
      "trg_length": preprocessing_config["trg_padding"],
      "src_padding_id": vocab.vocab["src"]["stoi"][VocabScratch.pad_token] if not hasattr(vocab, "tokenizer") else vocab.tokenizer.pad_token_id,
      "trg_padding_id": vocab.vocab["trg"]["stoi"][VocabScratch.pad_token] if not hasattr(vocab, "tokenizer") else vocab.tokenizer.pad_token_id,
      "src_unk_token": vocab.vocab["src"]["stoi"][VocabScratch.unk_token] if not hasattr(vocab, "tokenizer") else vocab.tokenizer.unk_token_id, 
      "trg_unk_token": vocab.vocab["trg"]["stoi"][VocabScratch.unk_token] if not hasattr(vocab, "tokenizer") else vocab.tokenizer.unk_token_id,
      "max_length": max_length
  }

def build_global_config(vocab, preprocessing_config, training_config, model_class, batch_size, epochs, use_copy, max_length, dataset_name, annotation, split_name):
  vocab_config = build_vocab_config(vocab, use_copy, max_length, preprocessing_config)
  model_config = model_class.create_config(**vocab_config)

  training_config["batch_size"] = batch_size
  training_config["epochs"] = epochs

  dataset_config = {
    "dataset_name": dataset_name,
    "annotation": annotation,
    "split_name": split_name,
  }

  return {"vocab": vocab_config, 
          "preprocessing": preprocessing_config, 
          "training": training_config, 
          "model": model_config,
          "dataset_config": dataset_config,
          "use_copy": use_copy,
          }

def save_config(folder, config):
  with open(f"{model_save_folder}/{folder}/config.json", "w") as file:
    file.write(json.dumps(config, indent=4))

def save_vocab(vocab, folder):
  if isinstance(vocab, dict):
    with open(f"{model_save_folder}/{folder}/vocab.json", "w") as file:
      file.write(json.dumps(vocab))
  elif isinstance(vocab, tuple):
    tokenizer, tok_to_spec, pref_to_spec = vocab
    tokenizer.save_pretrained(f"{model_save_folder}/{folder}/tokenizer")
    with open(f"{model_save_folder}/{folder}/token_to_special.json", "w") as file:
      file.write(json.dumps(tok_to_spec))
    with open(f"{model_save_folder}/{folder}/prefix_to_special.json", "w") as file:
      file.write(json.dumps(pref_to_spec))
  else:
    vocab.save_pretrained(f"{model_save_folder}/{folder}/tokenizer")

def load_vocab(folder, PretrainedTokenizer):
  if PretrainedTokenizer is not None:
    vocab = PretrainedTokenizer.from_pretrained(f"{model_save_folder}/{folder}/tokenizer")
    if PretrainedTokenizer == T5Tokenizer:
      with open(f"{model_save_folder}/{folder}/token_to_special.json") as file:
        token_to_special = json.load(file)
      with open(f"{model_save_folder}/{folder}/prefix_to_special.json") as file:
        prefix_to_special = json.load(file)
      vocab = (vocab, token_to_special, prefix_to_special)
    return vocab
  with open(f"{model_save_folder}/{folder}/vocab.json") as file:
    vocab = json.load(file)
  vocab["src"]["itos"] = {int(key): value for key, value in vocab["src"]["itos"].items()}
  vocab["trg"]["itos"] = {int(key): value for key, value in vocab["trg"]["itos"].items()}
  return vocab

def load_dataset(path):
  with open(path) as file:
    dataset = json.load(file)
  return dataset

def get_model_name(global_config_name):
  for model_name in ("bart", "t5", "transformer", "cnn"):
    if model_name in global_config_name:
      return model_name

def get_global_config_name(model_name, use_copy, dataset_name, dataset_annotation, split_name, run):
  if split_name != "original":
    return "_".join([model_name, "no_" * (not(use_copy)) + "copy", dataset_name, dataset_annotation, str(run), split_name])
  return "_".join([model_name, "no_" * (not(use_copy)) + "copy", dataset_name, dataset_annotation, str(run)])

def get_model(global_config_name, device, model_key="best-model"):
  with open(f"{model_save_folder}/{global_config_name}/config.json") as file:
    global_config = json.load(file)

  model_name = get_model_name(global_config_name)
  model_class = model_types[model_name]
  use_copy = global_config.get("use_copy", "no_copy" not in global_config_name)
  return model_class.load(folder=global_config_name, 
                            name=model_key, 
                            config=global_config["model"], 
                            use_copy=use_copy,
                            device=device,
                            model_save_folder=model_save_folder)

def remove_already_translated(dataset, gcn):
  previous_reports = {}
  if os.path.isfile(f"reports/{gcn}/report_temp.json"):
    with open(f"reports/{gcn}/report_temp.json") as file:
      previous_reports = json.load(file)
  print(len(previous_reports))
  done_ids = {entry["id"] for entry in previous_reports}
  print(len(done_ids))
  print(len(dataset))
  dataset = [entry for entry in dataset if entry["id"] not in done_ids]
  print(len(dataset))
  return dataset

def delete_run(gcn):
  shutil.rmtree(f"{model_save_folder}/{gcn}", ignore_errors=True)
  shutil.rmtree(f"reports/{gcn}", ignore_errors=True)

def get_vars(model_name, use_copy, dataset_name, dataset_annotation, split_name, run):
  # Use copy
  if use_copy in ("True", "False"): 
    use_copy = eval(use_copy)
  # Global config name
  gcn = get_global_config_name(model_name, use_copy, dataset_name, dataset_annotation, split_name, run)
  # Model class
  model_class = model_types[model_name]
  # Does the run need to be initialized
  init_run = gcn not in os.listdir(f"{model_save_folder}")
  # Device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return use_copy, gcn, model_class, init_run, device

def load_everything(gcn, dataset_name, dataset_annotation, model_name, use_copy, split_name, run, model_class, device, ref):
  with open(f"{model_save_folder}/{gcn}/config.json") as file:
      global_config = json.load(file)

  # If the given config has already finished and been tested
  need_train = global_config["training"]["state"] != "finished"
  need_test = gcn not in os.listdir(f"reports") or "report.json" not in os.listdir(f"reports/{gcn}")

  if need_train or need_test:
    vocab = get_vocab(gcn, model_name)
    dataset = load_dataset(f"data/datasets/{dataset_name}_dataset.json")
    Vocab.rework_dataset(dataset, dataset_annotation)
    batch_size = global_config["training"]["batch_size"]
    dataloaders = get_data(model_name, use_copy, dataset_name, dataset_annotation, split_name, run, dataset, batch_size, vocab, ref)
    model = model_class.load(folder=gcn, name="current-model", config=global_config["model"], use_copy=use_copy, device=device, model_save_folder=model_save_folder)
  else:
      dataset, vocab, dataloaders, model = None, None, (None, None, None), None
                              
  return global_config, need_train, need_test, dataset, vocab, dataloaders, model

def init_everything(gcn, dataset_name, dataset_annotation, model_name, use_copy, split_name, run, model_class, device, ref):
  need_train = True
  need_test = True
  # Initialize batch size
  batch_size = model_configs[model_name]["batch_size"][dataset_name]
  # Initialize max length
  max_length = 150
  if dataset_name == "lcquad2" and dataset_annotation == "linked":
    max_length = 200
  # Initialize preprocessing config
  preprocessing_config = {"min_freq": -1, "max_vocab_size": int(1e9), 
                          "special_tokens": VocabScratch.special_tokens, "src_padding": max_length-2, "trg_padding": max_length-2, "ref": ref}
  # Create vocabulary
  vocab = get_vocab(gcn, model_name)
  # Load dataset
  dataset = load_dataset(f"data/datasets/{dataset_name}_dataset.json")
  Vocab.rework_dataset(dataset, dataset_annotation)
  # Create dataloaders
  dataloaders = get_data(
                          model_name, 
                          use_copy, 
                          dataset_name, 
                          dataset_annotation, 
                          split_name, 
                          run, 
                          dataset, 
                          batch_size, 
                          vocab,
                          ref
                          )

  # Initialize epoch number
  epochs = model_configs[model_name]["epochs"][dataset_name]
  # Initialize global contfig
  global_config = build_global_config(vocab, 
                                      preprocessing_config, 
                                      training_config, 
                                      model_class, 
                                      batch_size, 
                                      epochs, 
                                      use_copy, 
                                      max_length,
                                      dataset_name,
                                      dataset_annotation,
                                      split_name)

  # Create model
  model = model_class.create_model(global_config["model"], use_copy, device)
  if model_name in ("transformer", "cnn"): 
    model.apply(initialize_weights)

  # Save everything
  print("Saving new config")
  os.makedirs(f"{model_save_folder}/{gcn}",exist_ok=True)
  vocab.save(gcn)
  model.save(gcn, "current-model", model_save_folder=model_save_folder)
  save_config(gcn, global_config)
  save_training_metrics(gcn, {"val_loss":[], "train_loss": [], "val_bleu": [], "train_bleu": []})
  return global_config, need_train, need_test, dataset, vocab, dataloaders, model

def print_before_training(dataset, dataloaders, model, split_name, vocab, need_show):
  if not need_show: return
  split_key = "_".join([split_name, "set"]) if split_name != "original" else "set"

  print("--- Run overview: ---")
  print("- Split:", split_name)
  print(f"     train: {len([entry for entry in dataset if entry[split_key]=='train'])} | val: {len([entry for entry in dataset if entry[split_key]=='val'])} | test: {len([entry for entry in dataset if entry[split_key]=='test'])}")
  print("- Batch sizes:")
  print(f"     train: {dataloaders[0].batch_size} | val: {dataloaders[1].batch_size} | test: {dataloaders[2].batch_size}")
  print(f"- Vocab size: src: {vocab.get_dim('src', False)} | src full: {vocab.get_dim('src', True)} | trg: {vocab.get_dim('trg', False)} | trg full: {vocab.get_dim('trg', True)}")
  print("- A few data example")
  for i, set_type in enumerate(("Train", "Val", "Test")):
    print(f"{set_type}:")
    src, mask, trg, idx = next(iter(dataloaders[i]))
    for k in np.random.randint(dataloaders[i].batch_size, size=3):
      print(" ".join(vocab.get_tokens(src[k:k+1].numpy(), "src")[0]))
      print(" ".join(vocab.get_tokens(trg[k:k+1].numpy(), "trg")[0]))
      print(f"Mask: {mask[k].sum().item() > 1e-6} | index: {idx[k]}")
      for entry in dataset:
        if str(entry["id"]) == str(idx[k].item()):
          print(" ", entry["question_raw"])
          print(" ", entry["question_tagged"])
          print(" ", entry["question_linked"])
          print(" ", entry["query_raw"])
          print(" ", entry["query_tagged"])
      print()
  print("- Model optimizer:", model.optimizer)
  print("- Model clip:", model.clip)
  print("--- End overview ---")
  print()

def run_config(model_name, use_copy, dataset_name, dataset_annotation, split_name, run, return_model=False):
  print("Runing config")
  # Set variables
  
  ###
  ### Initializing and loading ###
  ###

  use_copy, gcn, model_class, init_run, device = get_vars(model_name, use_copy, dataset_name, dataset_annotation, split_name, run)
  
  # Set here for testing with reformulated or classic questions
  ref = True
  if ref: assert dataset_annotation in ("raw", "linked")

  # Launch run
  print(f"=== {gcn} ===")
  
  # Check if run is feasible
  if dataset_annotation == "raw" and use_copy:
    print("Impossible configuration")
    return 

  # If the run has already been initialized
  if not init_run:
    print("Load config")
    get_everything = load_everything
  # If the run is not initialized
  else:
    get_everything = init_everything

  global_config, need_train, need_test, dataset, vocab, dataloaders, model = get_everything(
                                                                                              gcn, 
                                                                                              dataset_name, 
                                                                                              dataset_annotation, 
                                                                                              model_name, 
                                                                                              use_copy, 
                                                                                              split_name, 
                                                                                              run,
                                                                                              model_class,
                                                                                              device,
                                                                                              ref
                                                                                              )
  train_dataloader, val_dataloader, test_dataloader = dataloaders
  ###
  ### Training and generation ###
  ###

  # If needed we start / continue training
  print_before_training(dataset, dataloaders, model, split_name, vocab, need_train or need_test)
  
  if need_train:
    if return_model: return model
    model.to(device)
    print("Training model")
    train(model, train_dataloader, val_dataloader, device, vocab, global_config["training"], folder = gcn)
  
  # Once the training is completed we evaluate the model
  if need_test:
    model = model_class.load(folder=gcn, name="best-model", config=global_config["model"], use_copy=use_copy, device=device, model_save_folder=model_save_folder)
    if gcn not in os.listdir("reports"): os.mkdir(f"reports/{gcn}")
    model = model.to(device)
    report = test(model, test_dataloader, device, vocab, global_config["vocab"]["max_length"], use_copy, folder = gcn)
    with open(f"reports/{gcn}/report.json", "w") as file:
      file.write(json.dumps(report))

  # If we already have a report, we load it to show its score
  else:
    with open(f"reports/{gcn}/report.json") as file:
      report = json.load(file)

  # We show the basic 
  print(f"Test bleu best model: {bleu_score([rep['prd'] for rep in report], [[rep['trg']] for rep in report])*100:.2f}% | Test query accuracy best model: {sum([rep['prd']==rep['trg'] for rep in report])/len(report)}")
  print("_"*20)

if __name__ == "__main__":
  print("Starting main")
  run_config(*sys.argv[1:])
