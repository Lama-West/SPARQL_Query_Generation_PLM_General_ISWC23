
try:
    from conv_model import CNNSeq2Seq
    from transformer_model import TransfSeq2Seq
    from bart_model import BARTSeq2Seq
    from t5_model import T5Seq2Seq

except:
    from scripts.conv_model import CNNSeq2Seq
    from scripts.transformer_model import TransfSeq2Seq
    from scripts.bart_model import BARTSeq2Seq
    from scripts.t5_model import T5Seq2Seq

from transformers import BartTokenizer, T5Tokenizer

### To set up our experiments' parameters, we use string indicators that are mapped to specific configurations

# The model class for each model name
model_types = {
    "cnn": CNNSeq2Seq,
    "transformer": TransfSeq2Seq,
    "bart": BARTSeq2Seq,
    "t5": T5Seq2Seq,
}

# Possible values of the string parameters

# Copy options
copy_options = [True, False]

# Dataset options
dataset_types = ["lcquad1", "monument", "dbnqa", "lcquad2"]

# Question annotation options
dataset_annotations = ["raw", "tagged", "linked"]

# Some variable hyper parameters: 
# - Tokenizer, pre-trained weights, and clip value depend on the model
# - Batch size and number of epochs depend on the model and the dataset
model_configs = {
    "cnn": {
        "batch_size": {"lcquad1": 32, "monument": 32, "dbnqa": 5, "lcquad2": 16},
        "epochs": {"lcquad1": 500, "monument": 500, "dbnqa": 50, "lcquad2": 150},
        "pretrained_tokenizer": None,
        "pretrained_path": None,
        "clip": 0.1
    },
    "transformer": {
        "batch_size": {"lcquad1": 32, "monument": 32, "dbnqa": 32, "lcquad2": 16, "lcquad2reduced":32},
        "epochs": {"lcquad1": 500, "monument": 500, "dbnqa": 50, "lcquad2": 150},
        "pretrained_tokenizer": None,
        "pretrained_path": None,
        "clip": 1
    },
    "bart": {
        "batch_size": {"lcquad1": 16, "monument": 16, "dbnqa": 5, "lcquad2": 8},
        "epochs": {"lcquad1": 200, "monument": 200, "dbnqa": 20, "lcquad2": 50},
        "pretrained_tokenizer": BartTokenizer,
        "pretrained_path": "facebook/bart-base",
        "clip": 1
    },
    "t5": {
        "batch_size": {"lcquad1": 16, "monument": 16, "dbnqa": 5, "lcquad2": 8},
        "epochs": {"lcquad1": 200, "monument": 200, "dbnqa": 20, "lcquad2": 50},
        "pretrained_tokenizer": T5Tokenizer,
        "pretrained_path": "t5-small",
        "clip": 1
    }
}


training_config = {"current_epoch": 0, "current_batch": 0, "state": "init"}

model_save_folder = "results"
