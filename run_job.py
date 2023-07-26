from subprocess import run
import os

sbatch_script = """#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=def-azouaq
#SBATCH --mail-user=samuel.reyd@polymtl.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --output {0}-{1}-{2}-{3}-{4}-{5}.out

./lunch_session.sh {0} {1} {2} {3} {4} {5}

"""


#SBATCH --time=1-00:00:00

run_configs = (
   #(model_name, use_copy, dataset_name, dataset_annotation, split_name, run),
    # Transformer
    ("transformer", False, "lcquad2", "raw", "original", 1),
    #("transformer", False, "lcquad2", "raw", "original", 2),
    #("transformer", False, "lcquad2", "raw", "original", 3),

    #("transformer", False, "lcquad2", "linked", "original", 1),    
    #("transformer", False, "lcquad2", "linked", "original", 2),
    #("transformer", False, "lcquad2", "linked", "original", 3),

    #("transformer", True, "lcquad2", "linked", "original", 1),
    #("transformer", True, "lcquad2", "linked", "original", 2),
    #("transformer", True, "lcquad2", "linked", "original", 3),

    # CNN
    #("cnn", False, "lcquad2", "raw", "original", 1),
    #("cnn", False, "lcquad2", "raw", "original", 2),
    #("cnn", False, "lcquad2", "raw", "original", 3),
    
    #("cnn", False, "lcquad2", "linked", "original", 1),
    #("cnn", False, "lcquad2", "linked", "original", 2),
    #("cnn", False, "lcquad2", "linked", "original", 3),
    
    #("cnn", True, "lcquad2", "linked", "original", 1),
    #("cnn", True, "lcquad2", "linked", "original", 2),
    #("cnn", True, "lcquad2", "linked", "original", 3),


    # T5
    #("t5", False, "lcquad2", "raw", "original", 1),
    #("t5", False, "lcquad2", "raw", "original", 2),
    #("t5", False, "lcquad2", "raw", "original", 3),
    
    #("t5", False, "lcquad2", "linked", "original", 1),
    #("t5", False, "lcquad2", "linked", "original", 2),
    #("t5", False, "lcquad2", "linked", "original", 3),
    
    #("t5", True, "lcquad2", "linked", "original", 1),
    #("t5", True, "lcquad2", "linked", "original", 2),
    #("t5", True, "lcquad2", "linked", "original", 3),

    # BART
    #("bart", False, "lcquad2", "raw", "original", 1),
    #("bart", False, "lcquad2", "raw", "original", 2),
    #("bart", False, "lcquad2", "raw", "original", 3),
    
    #("bart", False, "lcquad2", "linked", "original", 1),
    #("bart", False, "lcquad2", "linked", "original", 2),
    #("bart", False, "lcquad2", "linked", "original", 3),
    
    #("bart", True, "lcquad2", "linked", "original", 1),
    #("bart", True, "lcquad2", "linked", "original", 2),
    #("bart", True, "lcquad2", "linked", "original", 3),
)

def get_global_config_name(model_name, use_copy, dataset_name, dataset_annotation, split_name, run):
  if split_name != "original":
    return "_".join([model_name, "no_" * (not(use_copy)) + "copy", dataset_name, dataset_annotation, str(run), split_name])
  return "_".join([model_name, "no_" * (not(use_copy)) + "copy", dataset_name, dataset_annotation, str(run)])

for args in (run_configs):
    print(args)
    gcn = get_global_config_name(*args)
    if gcn not in os.listdir("results") or gcn not in os.listdir("reports") or "report" not in os.listdir(f"reports/{gcn}"):
        run(["sbatch"], input=sbatch_script.format(*args).encode('utf-8'))
