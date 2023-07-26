import json
import argparse, sys
import pandas as pd
from typing import Dict, List, Any
from torchtext.data.metrics import bleu_score

from tqdm import tqdm
import numpy as np
import time
from collections import Counter
import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context

try:
    from scripts.main import *
    from scripts.query_functions import *
except:
    from main import *
    from query_functions import *


# Helper functions

# Function to get the string indicator of the dataset and the annotation from the name of the experiment
def get_folder_infos(folder):
  d_name = ""
  for name in ("lcquad2reduced", "lcquad2", "lcquad1", "dbnqa"):
    if name in folder:
      d_name = name
      break
  annotation = ""
  for ann in ("tagged", "raw", "linked"):
    if ann in folder:
      annotation = ann
      break
  return d_name, annotation


def complete_query(query, key, refs, correct, i, N, endpoint, agent):
  if query in refs:
    return refs[query], i
    # entry["query_info"][key] = refs[query]

  if "<unk>" in query:
    return {
        "answer": None,
        "result": "<unk> in query...",
        "is_error": True,
        "is_positive": False,
        "query": query
    }, i
  if key not in ("trg", "gold") and not correct and not "limit" in query:
    query = query + " limit 50"
  query_infos = get_query_info(query, endpoint, agent)
  refs[query] = query_infos
  i += 1
  # a = '\b'*12
  # if i > 1:
  #   print(a, end='')
  # print(f"{i+1:0=5d}/{N:0=5d}\n", end="")
  return query_infos, i

def generate_final_queries(report, gold_queries, refs):
  n = {"trg":0, "gold":0, "prd":0}
  for entry in report:
    gold_query = gold_queries[str(entry["id"])]
    entry["query_trg"] = escape_query(" ".join(entry["trg"]).replace("<<", "").replace(">>", ""))
    entry["query_gold"] = escape_query(" ".join(gold_query).replace("<<", "").replace(">>", ""))
    entry["query_prd"] = escape_query(" ".join(entry["prd"]).replace("<<", "").replace(">>", ""))
    entry["gold"] = gold_query

    if "prd_topn_beams" in entry:
      entry["query_topn_beams"] = []
      for beam in entry["prd_topn_beams"]:
        if beam:
          # query = sub_special_tokens(query)
          query = escape_query(" ".join(beam).replace("<<", "").replace(">>", ""))
          entry["query_topn_beams"].append(query)
          if query not in refs and "<unk>" not in query:
            n["prd"] += 1
        else:
          entry["query_topn_beams"].append(None)

    
    for key, query in zip(("trg", "gold", "prd"),(entry["query_trg"], entry["query_gold"], entry["query_prd"])):
      if query not in refs and "<unk>" not in query:
        n[key] += 1
  print(n)
  return n

def complete_report(folder, refs, endpoint, agent, gold_queries):
  # Show how many queries to check
  with open(f"reports/{folder}/report.json") as file:
    report = json.load(file)

  n = generate_final_queries(report, gold_queries, refs)

  N = sum(n.values())
  i = 0
  t = time.time()
  with tqdm(total=N) as pbar:
    for entry in report:

      # greedy prediction request
      entry["query_info"] = {}
      for key, query in zip(("trg", "gold", "prd"), (entry["query_trg"], entry["query_gold"], entry["query_prd"])):
        entry["query_info"][key], i = complete_query(query, key, refs, query == entry["query_gold"], i, N, endpoint, agent)
        pbar.n = i
        pbar.refresh()

      entry["query_info"]["correct"] = entry["query_info"]["gold"]["answer"] == entry["query_info"]["prd"]["answer"]

      # beam search predictions requests
      if "query_topn_beams" in entry and entry["query_topn_beams"][0] is not None:
        query = entry["query_topn_beams"][0]
        query_infos, i = complete_query(query, "query_topn_beams", refs, query == entry["query_gold"], i, N, endpoint, agent)
        pbar.n = i
        pbar.refresh()

        # We search forthe first positive beam query if none we keep the most prebable one i.e. the first
        for q in entry["query_topn_beams"]:
          infos, i = complete_query(q, "query_topn_beams", refs, query == entry["query_gold"], i, N, endpoint, agent)
          if infos["is_positive"]: 
            query_infos = infos
            query = q
            break

        entry["query_beam"] = query
        entry["query_info"]["prd_beam"] = query_infos
        entry["query_info"]["correct"] = entry["query_info"]["correct"] | (entry["query_info"]["gold"]["answer"] == query_infos["answer"]) 

      # save refs
      if time.time() - t > 5 * 60:
        save_refs(refs, endpoint)
        t = time.time()

  # save refs in the end 
  save_refs(refs, endpoint)

  with open(f"reports/{folder}/report_full.json", "w") as file:
    file.write(json.dumps(report))

  return report

def get_prec_rec_entry(pred, gold, optimistic):
  if type(pred) == bool: pred = [pred]
  if type(gold) == bool: gold = [gold]
  if pred is None: pred = []
  if gold is None: gold = []
  if len(gold) == 0 and len(pred) == 0:
    return int(optimistic), int(optimistic)
  if len(gold) == 0 or len(pred) == 0:
    return 0, 0
  count = len(set(gold) & set(pred))
  return count / len(pred), count / len(gold)

def get_f1(prec, recall):
  if prec == 0 or recall == 0:
    return 0
  return 2*(prec*recall) / (prec + recall)

def compute_metrics(report, beam, cor):
  # Compute bleu for all entries
  # bleu = bleu_score([rep["query_info"]["prd" + "_beam"*beam]["query"].split() for rep in report], [[rep["query_info"]["gold"]["query"].split()] for rep in report])
  rep_pred = [[elt.replace("<<", "").replace(">>","") for elt in rep["prd"]] for rep in report]
  rep_gold = [[[elt.replace("<<", "").replace(">>","") for elt in rep["gold"]]] for rep in report]
  #bleu = bleu_score([rep["prd"] for rep in report], [[rep["gold"]] for rep in report])
  bleu = bleu_score(rep_pred, rep_gold)
  # Instanciate values for loop
  if cor: query_acc, answer_acc, prec, rec = (0,) * 4
  else: query_acc, answer_acc_opt, answer_acc_pes, prec_opt, prec_pes, rec_opt, rec_pes = (0,) * 7
  if beam: inv_rank = 0

  # Loop
  for rep in report:
    prd_query = rep["query_info"]["prd" + "_beam"*beam]["query"] 
    prd_answer = rep["query_info"]["prd" + "_beam"*beam]["answer"]
    gold_query = rep["query_info"]["gold"]["query"]
    gold_answer = rep["query_info"]["gold"]["answer"]

    # Compute each metric
    query_acc += (prd_query == gold_query) # Query Accuracy
    if cor:
      answer_acc += (prd_answer == gold_answer) # Corrected Answer Accuracy
      prec_entry, rec_entry = get_prec_rec_entry(prd_answer, gold_answer, True) # Corrected Precision@k and Recall@k
      prec += prec_entry
      rec += rec_entry
    else:
      answer_acc_opt += (prd_answer == gold_answer) # Optimistic Answer Accuracy
      answer_acc_pes += (prd_answer == gold_answer) and bool(gold_answer) # Pessimistic Answer Accuracy
      prec_entry_opt, rec_entry_opt = get_prec_rec_entry(prd_answer, gold_answer, True) # Optimistic Precision@k and Recall@k
      prec_opt += prec_entry_opt
      rec_opt += rec_entry_opt
      prec_entry_pes, rec_entry_pes = get_prec_rec_entry(prd_answer, gold_answer, False) # Pessimistic Precision@k and Recall@k
      prec_pes += prec_entry_pes
      rec_pes += rec_entry_pes
    if beam:
      inv_rank += 1/(rep["query_topn_beams"].index(rep["query_info"]["prd_beam"]["query"].replace(" limit 50", "")) + 1)

  # Compute mean metrics over the entries
  N = len(report)

  metrics = {
      "bleu": bleu,
      "query_acc": query_acc / N,
  }
  if cor:
    metrics["answer_acc"] = answer_acc / N
    metrics["answer_f1"] = get_f1(prec / N, rec / N)
  else:
    metrics["answer_acc_opt"] = answer_acc_opt / N
    metrics["answer_acc_pes"] = answer_acc_pes / N
    metrics["answer_f1_opt"] = get_f1(prec_opt / N, rec_opt / N)
    metrics["answer_f1_pes"] = get_f1(prec_pes / N, rec_pes / N)
  if beam:
    metrics["inverse_rank"] = inv_rank / N

  return metrics

def compute_all_metrics(recompute=False):
  all_metrics = {"metrics_greedy":{}, "metrics_greedy_cor":{}, "metrics_beam":{}, "metrics_beam_cor":{}, }
  for folder in tqdm(os.listdir("reports")):
    if "Store" in folder: continue
    if "metrics.json" in os.listdir(f"reports/{folder}") and not recompute: 
      continue
    if "report_full.json" not in os.listdir(f"reports/{folder}"): continue
    with open(f"reports/{folder}/report_full.json") as file:
      report = json.load(file)
    # all_metrics[folder] = {}
    all_metrics["metrics_greedy"][folder] = compute_metrics(report, beam=False, cor=False)
    all_metrics["metrics_greedy_cor"][folder] = compute_metrics(list(filter(lambda rep: bool(rep["query_info"]["gold"]["answer"]), report)), beam=False, cor=True)
    if "bart" in folder and "no_copy" in folder:
      all_metrics["metrics_beam"][folder] = compute_metrics(report, beam=True, cor=False)
      all_metrics["metrics_beam_cor"][folder] = compute_metrics(list(filter(lambda rep: bool(rep["query_info"]["gold"]["answer"]), report)), beam=True, cor=True)
    for metric_type in ("metrics_greedy", "metrics_greedy_cor", "metrics_beam", "metrics_beam_cor"):
        if folder in all_metrics[metric_type]:
            with open(f"reports/{folder}/{metric_type}.json","w") as file:
                file.write(json.dumps(all_metrics[metric_type][folder]))
  with open(f"all_metrics.json","w") as file:
    file.write(json.dumps(all_metrics))

# Show metrics
def get_config_infos(config_name):
  model = None
  for m in ("cnn", "transformer", "bart", "t5"):
    if m in config_name:
      model = m
  
  dataset = None
  for d in ("lcquad1", "lcquad2", "dbnqa"):
    if d in config_name:
      dataset = d

  use_copy = not "no_copy" in config_name
  
  split = None
  for s in ("uniform_split", "template_split", "kb_elts_split"):
    if s in config_name:
      split = s
  if split is None: split = "original"

  annotation = None
  for a in ("raw", "tagged", "linked"):
    if a in config_name:
      annotation = a

  run_id = None
  for r in np.arange(10).astype(str):
    if "_" + r in config_name:
      run_id = r

  return model, dataset, use_copy, split, annotation, run_id

def show_metrics(models=("bart","cnn","transformers"),
            datasets=("lcquad2","lcquad1","dbnqa"),
            copy_usages=(True,False),
            splits=("uniform_split","original", "kb_elts_split","template_split"),
            annotations=("raw","tagged", "linked"),
            run=("1","2","3","4","5")
            ):
  
  full_df = pd.read_csv("full_results.csv").set_index("Unnamed: 0").rename_axis(None)
  show_df = full_df.drop([
                          # "bleu_prd", 
                          "valid_queries",
                          # "positive_queries",
                          "opt_prec",
                          "opt_recall",
                          "pes_prec",
                          "pes_recall",
                          "cor_prec",
                          "cor_recall",
                          # "answer_acc",
                          ], axis=1)
  show_df = show_df.rename(columns={'answer_acc_optimistic': 'ans_acc_opt',
                                    'answer_acc_pesimistic': 'ans_acc_pes',
                                    'answer_acc_no_empty': 'ans_acc_corr',
                                    'opt_f1': 'f1_opt',
                                    'pes_f1': 'f1_pes',
                                    'cor_f1': 'f1_corr',})
  for index in show_df.index:
    model, dataset, use_copy, split, annotation, run_id = get_config_infos(index)
    if (model not in models
        or dataset not in datasets
        or use_copy not in copy_usages
        or split not in splits
        or annotation not in annotations
        or run_id not in run):
      show_df = show_df.drop(index, axis=0)
  show_df = show_df.sort_values(by=["bleu", "ans_acc_corr"], ascending=False)
  try:
    display(show_df)
  except:
    print(show_df)

# Execute test
def make_run(folder, recompute_metrics, verbose, local=False):
  if verbose: print(folder)
  global report

  dataset_name, annotation = get_folder_infos(folder)

  # If no report continue
  if "report.json" not in os.listdir(f"reports/{folder}") and "report_full.json" not in os.listdir(f"reports/{folder}"):
    if verbose: 
      print("No report yet...")
      print()
    return None

  # Correct annotation
  if annotation == "linked":
      annotation = "tagged"

  # If report full already generated
  gold_queries = None
  if "report_full.json" in os.listdir(f"reports/{folder}"):
    if verbose: print("load full report")
    with open(f"reports/{folder}/report_full.json") as file:
      report = json.load(file)
      
  # Else generate full report
  else:

    # Get endpoint
    if "lcquad2" in folder:
      endpoint = endpoint_wikidata
    else:
      if local:
        endpoint = endpoint_dbpedia_2016
      else:
        endpoint = endpoint_dbpedia

    # Get the agent
    refs = get_refs(endpoint)
    agent = change_agent()

    # Get dataset for gold queries
    with open(f"data/datasets/{dataset_name}_dataset.json") as file:
      dataset = json.load(file)
      # print(dataset[0])
    
    # Make report
    gold_queries = {str(entry["id"]): entry["query_"+annotation].split(" ") for entry in dataset}
    report = complete_report(folder, refs, endpoint, agent, gold_queries)

  # Compute metrics
  # if not recompute_metrics and "metrics.json" in os.listdir(f"reports/{folder}"):
  #   with open(f"reports/{folder}/metrics.json") as file:
  #     metrics = json.load(file)
  # else:
  #   metrics = compute_metrics(report)
  #   with open(f"reports/{folder}/metrics.json", "w") as file:
  #     file.write(json.dumps(metrics))

  # Print metrics
  # if verbose:
  #   for m in metrics:
  #     if type(metrics[m]) is int:
  #       print(f"{m}: {metrics[m]}", end = " | ")
  #     else:
  #       print(f"{m}: {metrics[m]*100:2.2f}%", end = " | ")
  #   print("\b\b\n")
  # return metrics

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--local", help="Choose to use endpoint 2016", type=lambda x: bool(int(x)), default=True)
    parser.add_argument("--folder", help="Select just one folder to run", type=str, default="")
    args=parser.parse_args()
    try: os.chdir("/home/karou/Documents/samuel/NMT")
    except: pass
    all_metrics = []
    verbose = False
    if args.folder:
      metrics = make_run(sys.argv[1], recompute_metrics=True, verbose=verbose, local=args.local)
      sys.exit(0)
    else:
      for i, folder in enumerate(os.listdir("reports")) if verbose else tqdm(list(enumerate(os.listdir("reports")))):
          print(f"=== {folder} ({i+1}/{len(os.listdir('reports'))}) ===")
          # metrics = make_run(folder, recompute_metrics=True, verbose=verbose, local=True)
          # for metric in metrics:
          #   print(metric, f"{metrics[metric]:.2f} | ", end="")
          #if 'bart' in folder or 'dbnqa' in folder: 
          #    print("No")
          #    continue
          if "report_full.json" not in os.listdir(f"reports/{folder}"):
              make_run(folder, recompute_metrics=True, verbose=verbose, local=args.local)
          print()
      compute_all_metrics(True)
