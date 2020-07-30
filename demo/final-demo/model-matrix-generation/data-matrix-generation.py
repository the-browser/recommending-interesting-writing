# import necessary libraries
import numpy as np
from scipy.sparse import csr_matrix
import sys

sys.path.append(
    "/users/rohan/news-classification/ranking-featured-writing/rankfromsets"
)
import os
import argparse
from data_processing.articles import Articles
from models.models import InnerProduct
import data_processing.dictionaries as dictionary
from pathlib import Path
import ujson as json
import scipy
import argparse
from tokenizers import BertWordPieceTokenizer


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def expand_path(string):
    return Path(os.path.expandvars(string))


# create and parse necessary arguments
parser = argparse.ArgumentParser(
    description="Train model on article data and test evaluation"
)

parser.add_argument(
    "--dict_dir",
    type=expand_path,
    required=True,
    help="This is required to load dictionaries",
)

parser.add_argument(
    "--data_output_dir",
    type=expand_path,
    required=True,
    help="Location to save mapped and filtered data.",
)

parser.add_argument(
    "--matrix_output_dir",
    type=expand_path,
    required=True,
    help="Location to save article word sparse matrix.",
)

parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="Indicates which dataset for demo this is.",
)

parser.add_argument(
    "--map_items",
    type=str2bool,
    required=True,
    help="Determine if dataset need to be tokenized, mapped, and filtered.",
)

parser.add_argument(
    "--data_path", type=expand_path, required=True, help="Location to actual json data."
)

parser.add_argument(
    "--tokenizer_file", type=str, help="Designate tokenizer source file.",
)
parser.add_argument(
    "--min_length", type=int, default=0, help="Minimum Length for Articles in Dataset",
)

parser.add_argument(
    "--days", type=int, help="How many days old articles can be.",
)

parser.add_argument(
    "--filter",
    type=str2bool,
    default=True,
    help="Whether to apply filtering to the raw dataset",
)

args = parser.parse_args()


data_path = Path(args.data_path)
# dictionaries
dict_dir = Path(args.dict_dir)
final_word_ids, final_url_ids, final_publication_ids = dictionary.load_dictionaries(
    dict_dir
)
print("Dictionaries loaded.")

# tokenize, map, and filter data
if args.map_items:

    # initialize tokenizer from BERT library
    tokenizer = BertWordPieceTokenizer(args.tokenizer_file, lowercase=True,)
    print("Tokenizer Initialized!")
    if args.filter:
        raw_dataset = Articles(data_path)
        print("Initial: ", len(raw_dataset))
        if args.days is not None:
            proper_data = raw_dataset.map_items(
                tokenizer,
                final_url_ids,
                final_publication_ids,
                filter=True,
                min_length=args.min_length,
                day_range=args.days,
            )
        else:
            proper_data = raw_dataset.map_items(
                tokenizer,
                final_url_ids,
                final_publication_ids,
                filter=True,
                min_length=args.min_length,
            )
    else:
        proper_data = Articles(data_path)
        proper_data.map_items(tokenizer, final_url_ids, final_publication_ids)
    # save mapped data for easier access next iteration
    data_path = Path(args.data_output_dir)
    if not data_path.is_dir():
        data_path.mkdir()
    ending = "mapped_dataset_" + args.dataset_name + ".json"
    train_mapped_path = data_path / ending
    with open(train_mapped_path, "w") as file:
        json.dump(proper_data, file)
    raw_data = Articles(train_mapped_path)
    print("Final: ", len(raw_data))
    print(f"Filtered, Mapped Data saved to {data_path} directory")
    print("-------------------")

else:
    raw_data = Articles(data_path)
    print("Length: ", len(raw_data))

# generate sparse matrix with word ids for each article
rows = []
cols = []
for idx, item in enumerate(raw_data.examples):
    word_ids = list(set(item["text"]))
    number_of_words = np.arange(len(word_ids))
    rows.append(np.array(np.ones_like(number_of_words) * idx))
    cols.append(np.array(word_ids, dtype=np.int32))

final_rows = np.concatenate(rows, axis=None)
final_cols = np.concatenate(cols, axis=None)
final_data = np.ones_like(final_cols)

word_articles = csr_matrix(
    (final_data, (final_rows, final_cols)), shape=(len(raw_data), len(final_word_ids))
)
print("Article Matrix Created!")

ending = "csr_articles_" + args.dataset_name + ".npz"
matrix_path = args.matrix_output_dir / ending
scipy.sparse.save_npz(matrix_path, csr_matrix(word_articles))
print("Article-Word Matrix Saved")
