#!/bin/bash

conda create -n lm-critic python=3.8 cudatoolkit=10.1
conda activate lm-critic
pip install torch==1.6.0 torchvision==0.7.0
pip install transformers==4.3.3 datasets==1.3.0 absl-py rouge-score
pip install nltk wandb editdistance spacy==3.0.5
python3 -m nltk.downloader punkt
