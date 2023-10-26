# Scripts

This folder contains the following scripts:

- bart_model.py: the BartModel class 
- t5_model.py: the T5Model class
- conv_model.py: the CnnSeq2Seq class
- transformer_model.py: the TransformerModel class
- consts.py: the constant values such as hyper-parameters for each experiment
- data.py: the classes for tokenization depending on model type and annotation type
- train.py: the utility functions for main.py
- main.py: the main function to train a model and generate the answers on the test set
- query_functions.py: the utility functions for test_models.py
- test_models.py: the functions used to evaluate the output of a model as generated by the function of main.py
- split.py: the functions used to build a dataset split