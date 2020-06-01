# import necessary libraries
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
import sampling.sampler_util as sampler_util
import training.eval_util as eval_util
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import json
from pandas import json_normalize


# get arguments for script and parse
def expand_path(string):
    return Path(os.path.expandvars(string))


parser = argparse.ArgumentParser(description='Train model on article data and test evaluation')
parser.add_argument('--model_path',
                    type=expand_path,
                    help="This is required to load model.")

parser.add_argument('--dict_dir',
                    type=expand_path,
                    help="This is required to load dictionaries")

parser.add_argument('--dataset_path',
                    type=expand_path,
                    required=True,
                    help='Path to data to be ranked.')

args = parser.parse_args()

dict_dir = Path(args.dict_dir)
final_word_ids, final_url_ids, final_publication_ids = dictionary.load_dictionaries(dict_dir)
print("Dictionaries loaded.")

data_path = Path(args.dataset_path)
dataset = Articles(data_path)
print("Data loaded.")

dataset.tokenize()
print("Data tokenized.")
word_counter = collections.Counter()
for example in dataset.examples:
    word_counter.update(example['text'])

unique_words = [word for word in word_counter.keys()]
len(set(unique_words))

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

publication_emb = model.publication_embeddings.weight.data[0].cpu().numpy()
publication_bias = model.publication_bias.weight.data[0].cpu().numpy()
word_emb = model.attribute_emb_sum.weight.data.cpu().numpy()
word_bias = model.attribute_bias_sum.weight.data.cpu().numpy()

unique_words = list(set(unique_words))
word_emb_and_bias_dict = {}
for word in unique_words:
    if final_word_ids.get(word, 'None') != 'None':
        idx = final_word_ids.get(word, 'None')
        current_emb = list(word_emb[idx].astype(float))
        current_bias = list(word_bias[idx].astype(float))[0]
        current_short_dict = {'embedding': current_emb,
                              'bias': current_bias}
        word_emb_and_bias_dict[word] = current_short_dict

with open("word_to_emb+bias_dict.json", 'w') as file:
    json.dump(word_emb_and_bias_dict, file, separators=(',', ':'))
print("Model Embeddings and Bias Saved!")

pub_dict = {"embedding": list(publication_emb.astype(float)),
            "bias": list(publication_bias.astype(float))[0]}
with open("pub_emb+bias.json", 'w') as file:
    json.dump(pub_dict, file, separators=(',', ':'))
print("Publication Embeddings Saved!")

df = json_normalize(dataset)
df.drop(columns=['link', 'model_publication'], inplace=True)
df = df[df.text.apply(lambda x: len(x) > 400)]
df.to_json("select_demo_articles.json", orient='records')
print("Demo Articles Saved!")
