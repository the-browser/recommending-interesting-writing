# import necessary libraries
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import sys

sys.path.append(
    "/users/rohan/news-classification/ranking-featured-writing/rankfromsets"
)
import os
import argparse
from data_processing.articles import Articles
from models.models import InnerProduct
import data_processing.dictionaries as dictionary
import scipy
import json
import pandas as pd
import time

# get arguments for script and parse
def expand_path(string):
    return Path(os.path.expandvars(string))


parser = argparse.ArgumentParser(
    description="Train model on article data and test evaluation"
)
parser.add_argument(
    "--model_matrix_dir",
    type=expand_path,
    required=True,
    help="This is required to load model matrices.",
)

parser.add_argument(
    "--data_matrix_path",
    type=expand_path,
    required=True,
    help="This is required to load data matrices",
)

parser.add_argument(
    "--dict_dir", type=expand_path, required=True, help="Path to data to be ranked."
)

parser.add_argument(
    "--list_output_dir",
    type=expand_path,
    required=True,
    help="The place to store the generated html.",
)

parser.add_argument(
    "--real_data_path",
    type=expand_path,
    required=True,
    help="Mapped and filtered data to generate html with.",
)

parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="Indicates which dataset for demo this is.",
)

parser.add_argument(
    "--amount", type=int, default=75, help="Quantity of articles to include in list!"
)

parser.add_argument(
    "--emb_size", type=int, default=10, help="Embedding Size of Model Used"
)

args = parser.parse_args()

# load dictionaries
dict_dir = Path(args.dict_dir)
final_word_ids, final_url_ids, final_publication_ids = dictionary.load_dictionaries(
    dict_dir
)
print("Dictionaries loaded.")

# load numeric data for calculations
publication_emb = np.asarray(
    [
        0.77765566,
        0.76451594,
        0.75550663,
        -0.7732487,
        0.7341457,
        0.7216135,
        -0.7263404,
        0.73897207,
        -0.720818,
        0.73908365,
    ],
    dtype=np.float32,
)
publication_bias = 0.43974462

word_article_path = args.data_matrix_path
word_articles = scipy.sparse.load_npz(word_article_path)

word_emb_path = args.model_matrix_dir / "word_emb.npy"
word_emb = np.load(word_emb_path)
word_bias_path = args.model_matrix_dir / "word_bias.npy"
word_bias = np.load(word_bias_path)

# perform mathematical calculations to get default top predictions
print(word_articles.shape)
print(word_emb.shape)
time1 = time.time()
article_embeddings = word_articles.dot(word_emb)

emb_times_publication = np.dot(
    article_embeddings, publication_emb.reshape(args.emb_size, 1)
)

article_bias = word_articles.dot(word_bias)

product_with_bias = emb_times_publication + article_bias

word_counts = word_articles.sum(axis=1).reshape(word_articles.shape[0], 1)

final_logits = np.divide(product_with_bias, word_counts) + float(publication_bias)
time2 = time.time()
print(time2 - time1)

indices = final_logits.argsort(axis=0)[-args.amount :].reshape(args.amount)

word_logits = np.dot(word_emb, publication_emb.reshape(args.emb_size, 1)) + word_bias

top_articles = word_articles[indices.tolist()[0]]

broadcasted_words_per_article = top_articles.toarray() * word_logits.T

sorted_word_indices = broadcasted_words_per_article.argsort(axis=1)

return_articles = []

raw_data = Articles(args.real_data_path)
print(len(raw_data))
id_to_word = {v: k for k, v in final_word_ids.items()}

i = 0
for idx in indices.tolist()[0]:
    current_article = raw_data[int(idx)]
    current_article["logit"] = float(final_logits[int(idx)])
    current_sorted_words = sorted_word_indices[i]
    top_words = []
    least_words = []
    for top_word in current_sorted_words[-20:]:
        word = id_to_word[top_word]
        if "unused" not in word and "##" not in word and len(word) > 1:
            top_words.append(word)
    for least_word in current_sorted_words[:20]:
        word = id_to_word[least_word]
        if "unused" not in word and "##" not in word and len(word) > 1:
            least_words.append(word)
    current_article["top_words"] = top_words
    current_article["least_words"] = least_words
    return_articles.append(current_article)
    i += 1

ordered_return_articles = return_articles[::-1]
listed_df = pd.DataFrame(ordered_return_articles)
listed_df.drop(columns=["text", "url", "model_publication"], inplace=True)
ending = str(args.dataset_name) + "-top-list.csv"
final_html_path = args.list_output_dir / ending
listed_df.to_csv(final_html_path)

