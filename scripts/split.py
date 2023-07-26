import json
import re
import numpy as np
from collections import Counter
import time
import networkx as nx
import itertools
import time

def get_datasets():
  datasets = {}
  for name in ("lcquad1", "lcquad2"):
    with open(f"data/datasets/{name}_dataset.json") as file:
      datasets[name] = json.load(file)
  return datasets
datasets = get_datasets()

def get_words(entry, pattern, key="query_tagged"):
  return set(re.findall(pattern, entry[key]))

def get_template(entry):
  return set([entry["template_id"]])

ontology_pattern = "((?:db[opc]|wdt|rdfs?|p[qs]?):[^ >]+)"
URI_pattern = "((?:db[opcr]|wdt?|rdfs?|p[qs]?):[^ >]+)"
copy_pattern = "<<.*?>>"
resource_pattern = "((?:dbr|wd):[^ >]+)"



def make_split(G, n_instances, m, n):
  train_groups = set()
  train_size = 0
  test_size = 0
  for g in G:
    if train_size > n:
      assign_train = 0
    elif test_size > m:
      assign_train = 1
    else:
      p_train = (n-train_size)/(m+n-train_size-test_size)
      assign_train = np.random.binomial(n=1, p=p_train)
    if assign_train:
      train_groups.add(g)
      train_size += n_instances[g]
    else:
      test_size += n_instances[g]
  return train_groups

def get_rare_words(dataset, min_freq, pattern):
  all_words = []
  for entry in dataset:
    all_words += list(get_words(entry, pattern))
  counter = Counter(all_words)
  kept_words = set([word for word, freq in counter.items() if freq < min_freq])
  kept_entries = list()
  reserved_entries = list()
  entry2words = dict()
  for entry in dataset:
    if get_words(entry, pattern) & kept_words:
      kept_entries.append(entry["id"])
      entry2words[entry["id"]] = get_words(entry, pattern) & kept_words
    else:
      reserved_entries.append(entry["id"])
  return kept_words, kept_entries, reserved_entries, entry2words

def get_conected_components(entry2words):
  edges = []
  for entry1, entry2 in itertools.combinations(entry2words.keys(), 2):
    if entry2words[entry1] & entry2words[entry2]:
      edges.append((entry1,entry2))

  G = nx.Graph()
  G.add_nodes_from(list(entry2words.keys()))
  G.add_edges_from(edges)
  components = list(nx.connected_components(G))

  n_instances = dict()
  entry2component = dict()
  for i, component in enumerate(components):
    n_instances[i] = len(component)
    for entry in component:
      entry2component[entry] = i
  return n_instances, entry2component

def build_pattern_split(dataset, pattern, min_freq, train_prop=.8, N_repeat=500, seed=0):
  np.random.seed(seed)
  pre_processing_t0 = time.time()
  kept_words, kept_entries, reserved_entries, entry2words = get_rare_words(dataset, min_freq, pattern)
  n_instances, entry2component = get_conected_components(entry2words)
  pre_processing_t = time.time() - pre_processing_t0

  G = list(n_instances.keys())
  n_final = int(len(dataset)*train_prop)
  n = n_final - len(reserved_entries)
  m = len(dataset) - n_final
  splits = list()
  times = list()
  errors = list()
  for i in range(N_repeat):
    t0 = time.time()
    np.random.shuffle(G)

    train_groups = make_split(G, n_instances, m, n)

    train_size = sum([n_instances[g] for g in train_groups]) + len(reserved_entries)
    error = abs(train_size-n_final)/n_final
    errors.append(error)
    t = time.time() - t0
    times.append(t)
    splits.append(train_groups)
  print()
  print(f"Mean error: {np.mean(errors)*100:.2f}%")
  print(f"Best error: {np.min(errors)*100:.2f}%")
  print(f"Median error: {np.median(errors)*100:.2f}%")
  print(f"STD error: {np.std(errors)*100:.2f}%")
  print()
  if pre_processing_t > 60:
    print(f"Preprocessing time: {int(pre_processing_t//60)}mn{int(pre_processing_t%60)}s")
  else:
    print(f"Preprocessing time: {pre_processing_t:.2f}s")
  print(f"Mean time: {np.mean(times)*1000:.2f}ms")
  print(f"STD time: {np.std(times)*1000:.2f}ms")

def build_template_split(dataset, train_prop=.8, N_repeat=500, seed=0):
  np.random.seed(seed)
  n_instances = Counter([entry["template_id"] for entry in dataset])
  G = list(n_instances.keys())
  n = int(len(dataset)*train_prop)
  m = len(dataset) - n
  
  splits = list()
  times = list()
  errors = list()
  for i in range(N_repeat):
    t0 = time.time()
    np.random.shuffle(G)

    train_groups = make_split(G, n_instances, m, n)

    error = abs(sum([n_instances[g] for g in train_groups])-n)/n
    errors.append(error)
    t = time.time() - t0
    times.append(t)
    splits.append(train_groups)
  print(f"Mean error: {np.mean(errors)*100:.2f}%")
  print(f"Best error: {np.min(errors)*100:.2f}%")
  print(f"Median error: {np.median(errors)*100:.2f}%")
  print(f"STD error: {np.std(errors)*100:.2f}%")
  print()
  print(f"Mean time: {np.mean(times)*1000:.2f}ms")
  print(f"STD time: {np.std(times)*1000:.2f}ms")
