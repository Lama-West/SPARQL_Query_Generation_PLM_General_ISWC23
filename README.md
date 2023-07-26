
# SPARQL Query Generation
##### Git repository for SPARQL Query Generation using copy mechanism, question annotation and pre-trained models + study of models' generaliztion capabilities

## Files at root:
-  **demo.ipynb** Jupyter notebook for demonstration purposes. This notebook has section for training and evaluatind the model, and for creating new dataset splits
-  **lunch_session.sh** Bash script for launching a dynamic session on a SLURM server.
-  **run_job.py** Python file for running many experiments in parallel on a SLURM server.

## Folders:
### Data
##### Folder with the datasets. Each dataset is in a json file named *dataset_name*_dataset.json. Each json file is structure as one json entry being one dataset entry. Each entry contains the following keys:

* id: the id of the entry in our dataset
* original_question: the question as in the original dataset
* original_query: the query as in the original dataset
* question_raw: the question with pre-processing without annotation
* query_raw: the query with pre-processing
* question_tagged: the question in the tag-within setting
* query_tagged: the query with tag markers on KB elements
* question_linked: the question in the tag-end setting
* template_id: the id of the glabal template of the entry
* set: train/val/test for the original split
* template_split_set: train/val/test for the unknwon templates split
* all_KB_elements_split_set: train/val/test for the unknwon URIs split
* question_reformulated: reformulated question for the LC-QuAD datasets
* question_linked-ref: reformulated question for the LC-QuAD datasets with tag in the end
* answer_info: all info about the anwer of the question, i.e. 1) the query that was run to the KB, 2) the raw answer, boolean indicator of wether the 3) answer is empty or if 4) the query returned an error, and finally the 5) raw return of the query

### Scripts
##### Folder with all python files used to run the project. Contains the following scripts:
* bart_model.py: the BartModel class 
* t5_model.py: the T5Model class
* conv_model.py: the CnnSeq2Seq class
* transformer_model.py: the TransformerModel class
* consts.py: the constant values such as hyper-parameters for each experiment
* data.py: the classes for tokenization depending on model type and annotation type
* train.py: the utility functions for main.py
* main.py: the main function to train a model and generate the answers on the test set
* query_functions.py: the utility functions for test_models.py
* test_models.py: the functions used to evaluate the output of a model as generated by the function of main.py
* split.py: the functions used to build a dataset split
