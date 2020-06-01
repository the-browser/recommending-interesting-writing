# import necessary libraries
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import sys

sys.path.append(
    "/users/rohan/news-classification/ranking-featured-writing/bert-approach"
)
import os
import argparse
from data_processing.articles import Articles
from models.models import InnerProduct
import data_processing.dictionaries as dictionary
import scipy
import json


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
    help="This is required to load dictionaries",
)

parser.add_argument(
    "--dict_dir", type=expand_path, required=True, help="Path to data to be ranked."
)

parser.add_argument(
    "--html_output_dir",
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
        1.9955785,
        2.0590432,
        2.1761959,
        -2.0220742,
        1.9912238,
        -2.027111,
        -2.2134302,
        -1.9901487,
        2.0043654,
        -2.3093524,
        -2.0107682,
        2.002998,
        2.0343666,
        -2.0904067,
        2.4320478,
        2.0393133,
        2.0141761,
        1.992825,
        1.9798595,
        -2.0268648,
        -1.9863509,
        2.0398421,
        -1.9834707,
        -2.0138183,
        1.997613,
    ],
    dtype=np.float32,
)
publication_bias = 0.45260596

word_article_path = args.data_matrix_path
word_articles = scipy.sparse.load_npz(word_article_path)

word_emb_path = args.model_matrix_dir / "word_emb.npy"
word_emb = np.load(word_emb_path)
word_bias_path = args.model_matrix_dir / "word_bias.npy"
word_bias = np.load(word_bias_path)

# perform mathematical calculations to get default top predictions
print(word_articles.shape)
print(word_emb.shape)
article_embeddings = word_articles.dot(word_emb)

emb_times_publication = np.dot(article_embeddings, publication_emb.reshape(25, 1))

article_bias = word_articles.dot(word_bias)

product_with_bias = emb_times_publication + article_bias

word_counts = word_articles.sum(axis=1).reshape(word_articles.shape[0], 1)

final_logits = np.divide(product_with_bias, word_counts) + float(publication_bias)

indices = final_logits.argsort(axis=0)[-75:].reshape(75)

word_logits = np.dot(word_emb, publication_emb.reshape(25, 1)) + word_bias

top_articles = word_articles[indices.tolist()[0]]

broadcasted_words_per_article = top_articles.toarray() * word_logits.T

sorted_word_indices = broadcasted_words_per_article.argsort(axis=1)

return_articles = []

raw_data = Articles(args.real_data_path)
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


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# generate default html for the predictions
ordered_return_articles = return_articles[::-1]
grand_html = []

for idx, article in enumerate(ordered_return_articles):
    if not article["publication"] or len(article["publication"]) > 35:
        continue
    grand_html.append("<tr>")
    grand_html.append(
        f"<td class=\"title mdl-data-table__cell--non-numeric\"><a class=\"article_link\" href=\"{article['link']}\">{article['title']}</a>"
    )
    grand_html.append("<br></br>")
    grand_html.append("<br></br>")
    top_word_list = ""
    for item in article["top_words"]:
        if len(item) > 2 and not RepresentsInt(item):
            top_word_list += item
            top_word_list += ", "
    least_word_list = ""
    for item in article["least_words"]:
        if len(item) > 2:
            least_word_list += item
            least_word_list += ", "
    grand_html.append(f'<p class="top_words">{top_word_list[:-2]}</p>')
    grand_html.append(f'<p class="least_words">{least_word_list[:-2]}</p>')
    grand_html.append("</td>")
    grand_html.append(f"<td class=\"publication\">{article['publication']}</td>")
    grand_html.append(f"<td class=\"logit\">{article['logit']}</td>")
    grand_html.append("</tr>")

# save html to text file
ending = str(args.dataset_name) + ".txt"
final_html_path = args.html_output_dir / ending

with open(final_html_path, "w", encoding="utf-8") as file:
    file.write("\n".join(grand_html))
print("HTML default text saved!")
