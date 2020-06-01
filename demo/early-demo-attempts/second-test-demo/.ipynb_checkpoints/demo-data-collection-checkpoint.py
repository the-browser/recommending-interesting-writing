# import necessary libraries
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import sys
sys.path.append("..")
from models.models import InnerProduct
import pandas as pd
import torch
import collections
import numpy as np
import torch.nn as nn
import os
import argparse
from data_processing.articles import Articles
from models.models import InnerProduct
import data_processing.dictionaries as dictionary
from pathlib import Path
import json


# get arguments for script and parse
def expand_path(string):
    return Path(os.path.expandvars(string))


parser = argparse.ArgumentParser(description='Train model on article data and test evaluation')
parser.add_argument('--model_path',
                    type=expand_path,
                    required=True,
                    help="This is required to load model.")

parser.add_argument('--dict_dir',
                    type=expand_path,
                    required=True,
                    help="This is required to load dictionaries")

parser.add_argument('--dataset_path',
                    type=expand_path,
                    required=True,
                    help='Path to data to be ranked.')

parser.add_argument('--mapped_data_dir',
                    type=expand_path,
                    required=True,
                    help="The place to store the mapped data.")

args = parser.parse_args()

dict_dir = Path(args.dict_dir)
final_word_ids, final_url_ids, final_publication_ids = dictionary.load_dictionaries(dict_dir)
print("Dictionaries loaded.")

data_path = Path(args.dataset_path)
dataset = Articles(data_path)
print("Data loaded.")

abs_model_path = Path(args.model_path)
kwargs = dict(n_publications=len(final_publication_ids),
              n_articles=len(final_url_ids),
              n_attributes=len(final_word_ids),
              emb_size=100,
              sparse=False,
              use_article_emb=False,
              mode='mean')
model = InnerProduct(**kwargs)
model.load_state_dict(torch.load(abs_model_path))
print("Model Loaded.")

dataset.tokenize()
proper_data = dataset.map_items(final_word_ids,
                                final_url_ids,
                                final_publication_ids,
                                filter=True,
                                min_length=400)

data_path = Path(args.mapped_data_dir)
if not data_path.is_dir():
    data_path.mkdir()
mapped_data_path = data_path / "mapped-data"
if not mapped_data_path.is_dir():
    mapped_data_path.mkdir()
train_mapped_path = mapped_data_path / "mapped_dataset.json"
with open(train_mapped_path, "w") as file:
    json.dump(proper_data, file)
raw_data = Articles(train_mapped_path)
print("Final: ", len(raw_data))
print(f"Filtered, Mapped Data saved to {mapped_data_path} directory")
print("-------------------")

word_articles = csr_matrix((len(raw_data), len(final_word_ids)), dtype=np.float32).toarray()

for idx, item in enumerate(raw_data.examples):
    item['text'] = list(set(item['text']))
    for entry in item['text']:
        word_articles[idx][entry] = 1

publication_emb = model.publication_embeddings.weight.data[0].cpu().numpy()
publication_bias = model.publication_bias.weight.data[0].cpu().numpy()
word_emb = model.attribute_emb_sum.weight.data.cpu().numpy()
word_bias = model.attribute_bias_sum.weight.data.cpu().numpy()

np.save("word_articles.npy", word_articles)
print("Article-Word Matrix Saved")

np.save("word_emb.npy", word_emb)
print("Word Embeddings Saved")

np.save("word_bias.npy", word_bias)
print("Word Biases Saved")
