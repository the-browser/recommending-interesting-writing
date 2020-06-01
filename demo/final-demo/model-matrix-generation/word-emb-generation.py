# import necessary libraries
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import sys

sys.path.append(
    "/users/rohan/news-classification/ranking-featured-writing/rankfromsets"
)
from models.models import InnerProduct
import pandas as pd
import torch
import collections
import numpy as np
import torch.nn as nn
import os
import argparse
from data_processing.articles import Articles
import data_processing.dictionaries as dictionary
import json
from pandas import json_normalize
from scipy import sparse
import boto3
import argparse

# get arguments for script and parse
def expand_path(string):
    return Path(os.path.expandvars(string))


parser = argparse.ArgumentParser(
    description="Train model on article data and test evaluation"
)
parser.add_argument(
    "--model_path",
    type=expand_path,
    required=True,
    help="This is required to load model.",
)

parser.add_argument(
    "--dict_dir",
    type=expand_path,
    required=True,
    help="This is required to load dictionaries",
)

parser.add_argument(
    "--output_dir",
    type=expand_path,
    required=True,
    help="Location for model matrices to be saved.",
)

parser.add_argument(
    "--emb_size", type=int, required=True, help="Embedding dimension size of model."
)
args = parser.parse_args()

# load dictionaries for model parameters
dict_dir = Path(args.dict_dir)
final_word_ids, final_url_ids, final_publication_ids = dictionary.load_dictionaries(
    dict_dir
)
print("Dictionaries loaded.")

# pass in arguments for model and load model from saved state
abs_model_path = Path(args.model_path)
kwargs = dict(
    n_publications=len(final_publication_ids),
    n_articles=len(final_url_ids),
    n_attributes=len(final_word_ids),
    emb_size=args.emb_size,
    sparse=False,
    use_article_emb=False,
    mode="mean",
)

model = InnerProduct(**kwargs)
model.load_state_dict(torch.load(abs_model_path))
print("Model Loaded.")

# get model parameters
publication_emb = model.publication_embeddings.weight.data[0].cpu().numpy()
publication_bias = model.publication_bias.weight.data[0].cpu().numpy()
word_emb = model.attribute_emb_sum.weight.data.cpu().numpy()
word_bias = model.attribute_bias_sum.weight.data.cpu().numpy()

# save matrices data in numpy format for easy lambda loading and prediction generationn
emb_path = args.output_dir / "word_emb.npy"
np.save(emb_path, word_emb)
print("Word Embeddings Saved")

bias_path = args.output_dir / "word_bias.npy"
np.save(bias_path, word_bias)
print("Word Biases Saved")

reversed_word_ids = {v: k for k, v in final_word_ids.items()}
rev_path = args.output_dir / "reversed_word_ids.json"
with open(rev_path, "w") as file:
    json.dump(reversed_word_ids, file)
print("Reversed Id To Word Dictionary Saved.")
