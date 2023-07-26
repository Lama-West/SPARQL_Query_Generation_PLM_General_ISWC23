import numpy as np
import re
from collections import Counter
import torch
import os
import json
from torch.utils.data import TensorDataset, DataLoader
from typing import Union, Dict, List
from transformers import T5Tokenizer, BartTokenizer

try:
  from consts import *
except:
  from scripts.consts import *

# Origin class of vocabularies
class Vocab:

  # Remove from copy vocabulary the elements that are in the query but not in the question
  @staticmethod
  def rework_dataset(dataset, annotation):
    # Tagged case
    if annotation == "tagged":
      for entry in dataset:
        copyable_question = set(re.findall("<<.*?>>", entry["question_tagged"]))
        copyable_query = set(re.findall("<<.*?>>", entry["query_tagged"]))
        for resource in copyable_query - copyable_question:
          entry["query_tagged"] = entry["query_tagged"].replace(resource, resource[2:-2])

    # Linked case
    elif annotation == "linked":
      for entry in dataset:
        copyable_question = set(re.findall("<<.*?>>", entry["question_linked"]))
        copyable_query = set(re.findall("<<.*?>>", entry["query_tagged"]))
        for resource in copyable_query - copyable_question:
          entry["query_tagged"] = entry["query_tagged"].replace(resource, resource[2:-2])

  # First pre-processing step
  def process_entries(self):
    raise Exception("Undefined method")
  
  # Tokenize and vectorize question and queries: str -> list[int]
  def vectorize_entries(self):
    raise Exception("Undefined method")
  
  # Turn vectorized entries into dattaloaders
  def load_entries(self, entries, batch_size):
    # Turn the list [[question_ids1, mask1, query_ids1, id1], [question_ids2, mask2, query_ids2, id2], ...]
    # into the list [[question_ids1, query_ids2, ...], [mask1, mask2, ...], [query_ids1, query_ids2, ...], [id1, id2, ...]]
    question_ids, attention_masks, query_ids, ids = zip(*entries)
    # Make the list of ids into a tensor of integers
    ids = torch.LongTensor(ids)
    # Stack all other lists and put everything into a TensorDataset
    tensor_dataset = TensorDataset(torch.stack(question_ids), torch.stack(attention_masks), torch.stack(query_ids), ids)
    # Return the dataloader based on the dataset
    return DataLoader(tensor_dataset, shuffle=True, batch_size = batch_size)

  # Get the list of string tokens from a vector of indices
  def get_tokens(self):
    raise Exception("Undefined method")

class VocabScratch(Vocab):

  unk_token = "<unk>"
  sos_token = "<sos>"
  eos_token = "<eos>"
  pad_token = "<pad>"
  sep_token = "<sep>"
  special_tokens = [unk_token, sos_token, eos_token, pad_token, sep_token]

  def __init__(self, gcn):
    self.src_padding = None
    self.trg_padding = None
    if gcn in os.listdir(f"{model_save_folder}") and "vocab.json" in os.listdir(f"{model_save_folder}/{gcn}"):
      with open(f"{model_save_folder}/{gcn}/vocab.json") as file:
        self.vocab = json.load(file)
      self.vocab["src"]["itos"] = {int(key): value for key, value in self.vocab["src"]["itos"].items()}
      self.vocab["trg"]["itos"] = {int(key): value for key, value in self.vocab["trg"]["itos"].items()}
    else:
      self.vocab = None

  def __getitem__(self, key):
    return self.vocab[key]

  def save(self, folder):
    with open(f"{model_save_folder}/{folder}/vocab.json", "w") as file:
      file.write(json.dumps(self.vocab))

  def get_dim(self, vocab_type, full=True):
    if full:
      return len(self.vocab[vocab_type]["itos"])
    return len({word for word in self.vocab[vocab_type]["stoi"] if (
                                                                    "<<" not in word and
                                                                    ((":" not in word and 
                                                                    "'" not in word) or
                                                                    word in ("rdf:type", "rdfs:label", "wdt:P31")) and
                                                                    "not_a_resource" not in word)})

  def get_tokens(self, batch, vocab_type):
    output_tokens = []
    for pred_trg in batch:
      eos_ids = (pred_trg == self.vocab[vocab_type]["stoi"][self.eos_token]).nonzero()[0]

      if eos_ids.size:
        non_eos_tokens_ids = pred_trg[:eos_ids[0]]
      else:
        non_eos_tokens_ids = pred_trg
      
      output_tokens.append([self.vocab[vocab_type]["itos"][tok] if tok <= len(self.vocab[vocab_type]["itos"]) else self.unk_token for tok in non_eos_tokens_ids])
    return output_tokens

  def process_entries(self, dataset, question_key, query_key, set_key, annotation, use_copy, ref):
    if ref:
      if question_key == "question_raw":
        key_ref = "question_reformulated"
      else:
        key_ref = "question_linked-ref"
    else:
      key_ref = question_key
    return [{question_key: entry[key_ref], query_key: entry[query_key], "id": entry["id"], set_key: entry[set_key]} for entry in dataset]
  
  def vectorize_entries(self, entries, use_copy):
    questions, queries, ids = zip(*entries)

    questions_tokens = [question.split(" ") for question in questions]
    queries_tokens = [querie.split(" ") for querie in queries]

    if self.src_padding == None:
      self.src_padding = max(list(map(len, questions_tokens)))
    if self.trg_padding == None:
      self.trg_padding = max(list(map(len, queries_tokens)))
    if self.vocab is None:
      self.vocab = {}
      itos, stoi = self.get_vocab(questions_tokens)
      self.vocab["src"] = {"itos": itos, "stoi": stoi}
      itos, stoi = self.get_vocab(queries_tokens)
      self.vocab["trg"] = {"itos": itos, "stoi": stoi}
      if use_copy:
        self.rework_voc_for_copy()
    elif use_copy:
      additional_vocab = {}
      itos, stoi = self.get_vocab(questions_tokens)
      additional_vocab["src"] = {"itos": itos, "stoi": stoi}
      itos, stoi = self.get_vocab(queries_tokens)
      additional_vocab["trg"] = {"itos": itos, "stoi": stoi}
      self.update_reworked_vocab(additional_vocab)

    questions_vectorized = []
    query_vectorized = []
    for question_tokens, query_tokens in zip(questions_tokens, queries_tokens):
      src_vect = self.vectorize_tokens(question_tokens, self.vocab["src"]["stoi"], self.src_padding)
      trg_vect = self.vectorize_tokens(query_tokens, self.vocab["trg"]["stoi"], self.trg_padding)
      questions_vectorized.append( src_vect )
      query_vectorized.append( trg_vect )
    
    src_ids = torch.stack([torch.tensor(elt) for elt in questions_vectorized])
    trg_ids = torch.stack([torch.tensor(elt) for elt in query_vectorized])
    attention_masks = torch.empty(src_ids.shape) 
    return list(zip(src_ids, attention_masks, trg_ids, ids))

  def get_vocab(self, sentences):
    tokens = np.concatenate(sentences)
    voc = Counter(tokens)
    voc = dict(sorted(voc.items(), key=lambda x: x[1], reverse=True))
    words = self.special_tokens + list(voc.keys())
    itos = {i: s for i,s in enumerate(words)}
    stoi = {s: i for i,s in itos.items()}
    return itos, stoi

  def rework_voc_for_copy(self):
    english_syntax = [word for word in self.vocab["src"]["stoi"] if "<<" not in word]
    sparql_syntax = [word for word in self.vocab["trg"]["stoi"] if "<<" not in word]
    copy_vocab = set()
    copy_vocab |= {word for word in self.vocab["src"]["stoi"] if "<<" in word}
    copy_vocab |= {word for word in self.vocab["trg"]["stoi"] if "<<" in word}
    copy_vocab = list(copy_vocab)
    print(f"English: {len(english_syntax)} | SPARQL: {len(sparql_syntax)} | Copyable: {len(copy_vocab)}")
    #print(english_syntax)
    #print(sparql_syntax)
    min_voc = min([english_syntax, sparql_syntax], key=len)
    max_voc = max([english_syntax, sparql_syntax], key=len)
    min_voc += [f"<not_a_resource_{i}>" for i in range(len(max_voc) - len(min_voc))]
    src_voc = english_syntax + copy_vocab
    trg_voc = sparql_syntax + copy_vocab
    self.vocab["src"]["itos"] = {i: s for i,s in enumerate(src_voc)}
    self.vocab["src"]["stoi"] = {s: i for i,s in self.vocab["src"]["itos"].items()}
    self.vocab["trg"]["itos"] = {i: s for i,s in enumerate(trg_voc)}
    self.vocab["trg"]["stoi"] = {s: i for i,s in self.vocab["trg"]["itos"].items()}

  def update_reworked_vocab(self, additional_vocab):
    print(f"Adding new vocab to one of size {len(self.vocab['src']['itos'])}", end=" --> ")
    original_resources = {word for word in self.vocab["src"]["stoi"] if "<<" in word}
    additionnal_resource = {word for word in additional_vocab["src"]["stoi"] if "<<" in word} - original_resources
    for word in additionnal_resource:
      i = len(self.vocab["src"]["stoi"])
      self.vocab["src"]["stoi"][word] = i
      self.vocab["src"]["itos"][i] = word
      self.vocab["trg"]["stoi"][word] = i
      self.vocab["trg"]["itos"][i] = word
    print(f"new length: {len(self.vocab['src']['itos'])}")

  def vectorize_tokens(self, tokens, stoi, padding):
    vect_tokens = [stoi[self.sos_token]]
    for token in tokens[:padding]:
      vect_tokens.append(stoi.get(token, stoi[self.unk_token]))
    vect_tokens.append(stoi[self.eos_token])
    for i in range(len(tokens), padding):
      vect_tokens.append(stoi[self.pad_token])
    return vect_tokens

  def mask_unk_resources(self, question_ids, query_ids):
    M = max(question_ids)
    for i, id in enumerate(query_ids):
      if id > M:
        query_ids[i] = self.vocab["trg"]["stoi"][self.unk_token]
    return query_ids

class VocabPreTrained(Vocab):

  def __init__(self, gcn):
    for model_name in ("bart", "t5"):
      if model_name in gcn:
        break
    else:
      raise Exception("No pre-trained model name in the config")
    if gcn in os.listdir(f"{model_save_folder}") and "tokenizer" in os.listdir(f"{model_save_folder}/{gcn}"):
      self.tokenizer = self.tokenizer_class.from_pretrained(f"{model_save_folder}/{gcn}/tokenizer")
      with open(f"{model_save_folder}/{gcn}/token_to_special.json") as file:
        self.tok_to_spec = json.load(file)
      with open(f"{model_save_folder}/{gcn}/prefix_to_special.json") as file:
        self.pref_to_spec = json.load(file)
    else:
      try: 
        self.tokenizer = self.tokenizer_class.from_pretrained(model_configs[model_name]["pretrained_path"])
      except: 
        self.tokenizer = self.tokenizer_class.from_pretrained(model_name + "_tokenizer")
      self.tok_to_spec = None
      self.pref_to_spec = None
    self.unk_token_id = self.tokenizer.unk_token_id
    self.special_tokens = list(self.tokenizer.special_tokens_map.values())

  def get_tokens(self, batch, vocab_type):  
    sentences = [] 
    batch[batch < 0] = self.tokenizer.pad_token_id
    for sentence in self.tokenizer.batch_decode(batch):
      sentence = sentence.split("</s>")[0]
      sentence = re.sub("([^<]|^)<([^<]|$)", r"\1 <\2", sentence)
      sentence = sentence.replace("<s>", "<s> ")
      sentence = re.sub(" +", " ", sentence)
      sentence = self.sub_special_tokens(sentence)
      sentences.append(sentence.split())
    return sentences

  def get_dim(self, vocab_type, full=True):
    if full:
      return len(self.tokenizer)
    return len({word for word in self.tokenizer.get_vocab().keys() if "<<" not in word})

  def save(self, folder):
    self.tokenizer.save_pretrained(f"{model_save_folder}/{folder}/tokenizer")
    with open(f"{model_save_folder}/{folder}/token_to_special.json", "w") as file:
      file.write(json.dumps(self.tok_to_spec))
    with open(f"{model_save_folder}/{folder}/prefix_to_special.json", "w") as file:
      file.write(json.dumps(self.pref_to_spec))
         
  def process_entries(self, dataset, question_key, query_key, set_key, annotation, use_copy, ref):
    if self.tok_to_spec is None and self.pref_to_spec is None: 
      self.get_special_tokens_matching(dataset)
    # else:
    #   print(self.pref_to_spec)
    entries = dataset[:]
    if annotation == "raw": entries = self.process_raw(entries, question_key, query_key, set_key, use_copy, ref)
    elif annotation == "linked": entries = self.process_linked(entries, question_key, query_key, set_key, use_copy, ref)
    elif annotation == "tagged": entries = self.process_tagged(entries, question_key, query_key, set_key, use_copy)
    return entries
  
  def vectorize_entries(self, entries, use_copy):
    questions, queries, ids = zip(*entries)
    questions_dict = self.tokenizer(questions, truncation=True, max_length=100, padding="max_length", return_tensors="pt")
    queries_dict = self.tokenizer(queries, truncation=True, max_length=100, padding="max_length", return_tensors="pt")
    return list(zip(questions_dict["input_ids"], questions_dict["attention_mask"], queries_dict["input_ids"], ids))

  def process_linked(self, dataset, question_key, query_key, set_key, use_copy, ref):
    entries = []
    if use_copy:
      self._get_copyable_vocab(dataset)
    if ref:
        key_ref = "question_linked-ref"
    else:
        key_ref = question_key
    for entry in dataset:
      if not use_copy:
        # print(entry["question_linked"], entry["question_linked"].index("<sep>"))
        i = entry[key_ref].index("<sep>")
        question = entry[key_ref][:i-1]
        tags = entry[key_ref][i-1:]
        # entities = [entity[2:-2] for entity in entities.split(" ") if ":" in entity and not "'" in entity]
        # np.random.shuffle(entities)
        tags = tags.replace("<<", "").replace(">>", "")
        question = question + self.insert_special_tokens(tags)
      else:
        question = entry[key_ref]
      entries.append({
          question_key: question,
          query_key: self.insert_special_tokens(entry["query_tagged"] if use_copy else entry["query_raw"]),
          "id": entry["id"],
          set_key: entry[set_key]
      })
    return entries

  def process_tagged(self, dataset, question_key, query_key, set_key, use_copy):
    entries = []
    if use_copy:
      self._get_copyable_vocab(dataset)
    for entry in dataset:
      question = entry["question_tagged"] + " "
      if not use_copy:
        resources = re.findall("<<(.*?)>>", question)
        for resource in resources:
          if ":" in resource and "'" not in resource:
            question = question.replace(f" <<{resource}>> ", self.insert_special_tokens(" " + resource + " "))
          else:
            question = question.replace(f"<<{resource}>>", resource)
        
      entries.append({
          question_key: question.strip(),
          query_key: self.insert_special_tokens(entry["query_tagged"] if use_copy else entry["query_raw"]),
          "id": entry["id"],
          set_key: entry[set_key]
      })
    return entries

  def process_raw(self, dataset, question_key, query_key, set_key, use_copy, ref):
    if ref:
      key_ref = "question_reformulated"
    else:
      key_ref = "question_raw"
    return [{
        question_key: entry[key_ref],
        query_key: self.insert_special_tokens(entry["query_raw"]),
        "id": entry["id"],
        set_key: entry[set_key]
    } for entry in dataset]

  def _get_special_tokens_matching(self, dataset, additional_special_tokens):
    # print("Generating special tokens")
    sparql_shema_voc = set()
    for data in dataset:
      sparql_shema_voc |= set(re.sub("(<<.*?>>|(?:db[rocp]|wdt?|ps?):[^ ]+)", "", data["query_tagged"]).split())
    sparql_shema_voc.add("<sep>")
    resources = set()
    for data in dataset:
      resources |= set(re.findall("<<.*?>>", data["query_tagged"]))
    prefixes = set([elt.split(":")[0][2:] + ":" for elt in resources if ":" in elt and not "'" in elt])
    if additional_special_tokens is None: 
      sparql_schema_special_tokens = [f"<{token}>" if token != "<sep>" else "<sep>" for token in sparql_shema_voc]
      prefix_schema_special_tokens = [f"<{token}>" for token in prefixes]
      self.tokenizer.add_tokens(sparql_schema_special_tokens + prefix_schema_special_tokens)
    else:
      sparql_schema_special_tokens = additional_special_tokens[:len(sparql_shema_voc)]
      prefix_schema_special_tokens = additional_special_tokens[len(sparql_shema_voc):]
    self.tok_to_spec = dict(zip(sparql_shema_voc, sparql_schema_special_tokens))
    self.pref_to_spec = dict(zip(prefixes, prefix_schema_special_tokens))

  def _get_copyable_vocab(self, dataset):
    resources = set()
    for entry in dataset:
      resources |= set(re.findall("<<.*?>>", entry["question_tagged"]))
      resources |= set(re.findall("<<.*?>>", entry["query_tagged"]))
    print(len(resources))
    self.tokenizer.add_tokens(list(resources))

  def sub_special_tokens(self, query):
    for tok, spec in self.tok_to_spec.items():
      query = query.replace(spec, tok)
      # query = query.replace(f" {spec} ", f" {tok} ")
      # if query.startswith(spec) and not query.startswith("<"):
      #   query = tok + query[len(spec):]
      # if query.endswith(spec) and not query.endswith(">"):
      #   query = query[:-len(spec)] + tok

    for pref, spec in self.pref_to_spec.items():
      query = query.replace(" "+spec + " ", " "+pref)
      if query.startswith(spec) and not query.startswith("<"):
        query = pref + query[len(spec):]
    return query

  def insert_special_tokens(self, query):
    for tok, spec in self.tok_to_spec.items():
      query = query.replace(f" {tok} ", f" {spec} ")
      if query.startswith(tok) and not query.startswith("<"):
        query = spec + query[len(tok):]
      if query.endswith(tok) and not query.endswith(">"):
        query = query[:-len(tok)] + spec
    for pref, spec in self.pref_to_spec.items():
      query = query.replace(" "+pref, " "+spec)
      if query.startswith(pref) and not query.startswith("<"):
        query = spec + query[len(pref):]
    return query

  def mask_unk_resources(self, question_ids, query_ids):
    M = max(question_ids)
    for i, id in enumerate(query_ids):
      if id > M and id > self.get_dim(None, full=False):
        query_ids[i] = self.unk_token_id
    return query_ids

class VocabBart(VocabPreTrained):
  pretrained_path = "facebook/bart-base"
  tokenizer_class = BartTokenizer

  def get_special_tokens_matching(self, dataset):
    self._get_special_tokens_matching(dataset, None)

class VocabT5(VocabPreTrained):
  pretrained_path = "t5-small"
  tokenizer_class = T5Tokenizer

  def get_special_tokens_matching(self, dataset):
    try:
        special_tokens = self.tokenizer.get_sentinel_tokens()
    except:
        special_tokens = self.tokenizer.additional_special_tokens
    self._get_special_tokens_matching(dataset, special_tokens)

def get_data(model_name, use_copy, dataset_name, dataset_annotation, split_name, run, dataset, batch_size, vocab: Vocab, ref):
  question_key = "question_" + dataset_annotation
  query_key = "query_" + dataset_annotation if dataset_annotation != "linked" else "query_tagged"

  if split_name == "original": split_name = ""
  split_key = "_".join([split_name, "set"]) if split_name else "set"

  entries = vocab.process_entries(dataset, question_key, query_key, split_key, dataset_annotation, use_copy, ref)

  train_entries = [(entry[question_key], entry[query_key], entry["id"]) for entry in entries if entry[split_key]=="train"]
  val_entries   = [(entry[question_key], entry[query_key], entry["id"]) for entry in entries if entry[split_key]=="val"]
  test_entries  = [(entry[question_key], entry[query_key], entry["id"]) for entry in entries if entry[split_key]=="test"]

  train_vectorized = vocab.vectorize_entries(train_entries, use_copy)
  val_vectorized = vocab.vectorize_entries(val_entries, use_copy)
  test_vectorized = vocab.vectorize_entries(test_entries, use_copy)

  if use_copy:
    for vecorized_entries in (train_vectorized, val_vectorized, test_vectorized):
      for question_ids, attention_masks, query_ids, ids in vecorized_entries:
        vocab.mask_unk_resources(question_ids, query_ids)

  train_dataloader = vocab.load_entries(train_vectorized, batch_size)
  val_dataloader = vocab.load_entries(val_vectorized, batch_size)
  test_dataloader = vocab.load_entries(test_vectorized, batch_size)

  return train_dataloader, val_dataloader, test_dataloader

def get_vocab(gcn, model_name) -> Vocab:
  # if model_name == "t5": 
  #   return VocabT5(gcn)
  if model_name == "bart": 
    return VocabBart(gcn)
  elif model_name == "t5":
    return VocabT5(gcn)
  return VocabScratch(gcn)
