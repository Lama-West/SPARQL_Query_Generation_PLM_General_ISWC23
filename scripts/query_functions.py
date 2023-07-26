import json
import os
from json import JSONDecodeError

import json
import re
import numpy as np
from tqdm import tqdm
import time

from urllib.error import HTTPError
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed, EndPointInternalError
from http.client import IncompleteRead

endpoint_wikidata = "https://query.wikidata.org/sparql"
endpoint_dbpedia = "http://dbpedia.org/sparql"
endpoint_dbpedia_2016 = "http://127.0.0.1:8892/sparql"
endpoint_dbpedia_2018 = "http://127.0.0.1:8890/sparql"

prefix_endpoint_2016 = prefix = """PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX dbc: <http://dbpedia.org/resource/Category>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""

endpoint_names = {endpoint_wikidata: "wikidata", endpoint_dbpedia: "dbpedia", endpoint_dbpedia_2016: "dbpedia-2014"}

GET_RESOURCES_PURE_SPARQL_RE = re.compile("[^ {]+:[^ }]+")

# Query functions

def query_sparql(query, endpoint, agent='example-EX (https://example.com/; mail@example.com)'):
  if endpoint in (endpoint_dbpedia_2016, endpoint_dbpedia_2018):
    query = prefix_endpoint_2016 + query
  #print(query)
  #print(endpoint)
  sparql = SPARQLWrapper(endpoint, agent=agent)
  sparql.setReturnFormat(JSON)

  sparql.setQuery(query)

  return sparql.query().convert()
  
def change_agent():
  code = ''
  for i in np.random.randint(65, 65+24, size=5):
    code = code + chr(i)
  agent = f'example-{code} (https://example.com/; mail@example.com)'
  return agent
  

def get_results(result):
  if "results" in result:
    ret = []
    for res in result["results"]["bindings"]:
      for value in res.values():
        ret.append(value["value"])
    return ret
  if "boolean" in result:
    return result["boolean"]
    
def get_query_info(query, endpoint, agent):
  query_info = {}
  try:
    result = query_sparql(query, endpoint, agent)
    answer = get_results(result)
    is_positive = bool(answer)
    is_error = False
  except (HTTPError, QueryBadFormed, EndPointInternalError, JSONDecodeError, ValueError, IncompleteRead) as error:
    if error.args:
      result = str(error.args[0])
    else:
      result = str(error)
    if result.split(":")[0] == "HTTP Error 403":
      agent = change_agent()
    answer = None
    is_positive = False
    is_error = True
  query_info["query"] = query
  query_info["result"] = result
  query_info["answer"] = answer
  query_info["is_positive"] = is_positive
  query_info["is_error"] = is_error
  return query_info
   
def test_dataset_queries(dataset, refs, endpoint, query_key):
  agent = change_agent()
  errors = {}
  negatives = {}
  t = time.time()
  for entry in tqdm(dataset):
    query = entry[query_key].replace("<<", "").replace(">>", "")
    query = escape_query(query)
    if query in refs:
      query_info = refs[query]
    else:
      query_info = get_query_info(query, endpoint, agent)
    #if not query_info["is_error"]:
      refs[query] = query_info
    #else:
    if query_info["is_error"]:
      errors[query] = query_info
    if not query_info["is_positive"]:
      negatives[query] = query_info
    if time.time() - t > 5 * 60:
      t = time.time()
      save_refs(refs, endpoint)
  save_refs(refs, endpoint)
  return errors, negatives
  
def get_refs(endpoint):
    if not os.path.isfile(f"data/sparql_refs_{endpoint_names[endpoint]}.json"):
        refs = dict()
        with open(f"data/sparql_refs_{endpoint_names[endpoint]}.json", "w") as file:
            file.write(json.dumps(refs))
    else:
        with open(f"data/sparql_refs_{endpoint_names[endpoint]}.json") as file:
            refs = json.load(file)
    return refs
  
def save_refs(refs, endpoint):
  with open(f"data/sparql_refs_{endpoint_names[endpoint]}.json", "w") as file:
    file.write(json.dumps(refs))
  
# Escape functions
def escape_in_resource(match, char):
  resource = match.group(0)
  resource = resource.replace(char, '\\'+char)
  return resource

def escape_char(query, char):
  query = GET_RESOURCES_PURE_SPARQL_RE.sub(lambda x: escape_in_resource(x, char), query)
  return query

def escape_ampersands(query: str) -> str:
    amp = query.find('&')
    while amp > 0:
        if query[amp - 1] != '&' and query[amp + 1] != '&':
            query = query[:amp] + '\\' + query[amp:]
        amp = query.find('&', amp + 2)
    return query

def escape_plus(query: str) -> str:
    idx = query.find('+')
    while idx > 0:
        query = query[:idx] + '\\' + query[idx:]
        idx = query.find('+', idx + 2)
    return query

def escape_star(query: str) -> str:
    idx = query.find('*')
    while idx > 0:
        query = query[:idx] + '\\' + query[idx:]
        idx = query.find('*', idx + 2)
    return query

def escape_decimals(query):
  query = re.sub("([0-9]) . ([0-9])", r"\1.\2", query)
  query = re.sub("e - ([0-9])", r"e-\1", query)
  query = re.sub("e \\\\\+ ([0-9])", r"e+\1", query)
  return query

def delete_quote_spaces(query):
  query = re.sub(" ' (.*?) ' ", r" '\1' ", query)
  return query

def escape_query(query: str) -> str:
    query = escape_char(query, "(")
    query = escape_char(query, ")")
    query = escape_char(query, ",")
    query = escape_char(query, ".")
    query = escape_char(query, "'")

    query = query.replace("as k ", "ask ")

    query = escape_ampersands(query)
    query = escape_plus(query)
    query = escape_star(query)
    
    query = query.replace("!", "\\!")
    query = query.replace("/", "\\/")

    query = delete_quote_spaces(query)
    query = escape_decimals(query)
    return query
