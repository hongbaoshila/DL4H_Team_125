# DL4H_Team_125

1. Set up Language and Module based on the original paper (https://github.com/zdy93/FTL-Trans/blob/master/README.md)

Python3 == 3.x.x

Module

torch==1.3.1+cu92

pytorch-pretrained-bert==0.6.2

pytorch-transformers==1.2.0

tqdm==4.37.0

dotmap==1.3.8

six==1.13.0

matplotlib==3.1.1

numpy==1.17.3

pandas==0.25.3


2. Download MIMIC-III data from https://mimic.mit.edu/docs/gettingstarted/


3. Run preprocessecoli.py by uncommenting the code part by part to process the data step by step to prevent out of memory issue.


4. Run run_bert_am.py and run_clbert_tlstm.py. Adjust the parameters as needed.
