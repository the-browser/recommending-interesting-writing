import os
import argparse
import pathlib


def expand_path(string):
    return pathlib.Path(os.path.expandvars(string))


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def add_data(parser):
    parser.add_argument(
        "--dataset_path",
        type=expand_path,
        required=True,
        help="Path to data to be ranked.",
    )
    parser.add_argument(
        "--data_dir",
        type=expand_path,
        required=False,
        help="Path for data storage and loading. Only required if dictionaries and/or mapped-data being created.",
    )
    parser.add_argument(
        "--output_dir",
        type=expand_path,
        required=True,
        help="Directory for storing results",
    )
    parser.add_argument(
        "--dict_dir", type=expand_path, help="Path where dictionaries are located."
    )
    parser.add_argument(
        "--map_items",
        type=str2bool,
        default=True,
        help="True if data must be mapped to ids, else False",
    )
    parser.add_argument(
        "--tokenize",
        type=str2bool,
        default=True,
        help="Tokenize and split data fields.",
    )
    parser.add_argument(
        "--tokenizer_file",
        type=str,
        default="some_file_path.txt",
        help="Designate tokenizer source file.",
    )
    parser.add_argument(
        "--min_article_length",
        type=int,
        default=100,
        help="Minimum article length to generate predictions for.",
    )
    parser.add_argument(
        "--days_old", type=int, help="Maximum age of the article in days",
    )


def add_model(parser):
    parser.add_argument(
        "--model_path", type=expand_path, help="This is required to load model."
    )
    parser.add_argument(
        "--use_all_words",
        type=str2bool,
        default=False,
        help="Use all words in article for model",
    )
    parser.add_argument(
        "--words_to_use",
        type=int,
        default=400,
        help="Number of words to use from article",
    )
    parser.add_argument(
        "--use_gpu", type=str2bool, default=True, help="Use GPU for predictions"
    )
    parser.add_argument(
        "--emb_size",
        type=int,
        default=100,
        help="Set word and publication embedding size",
    )
    parser.add_argument(
        "--word_embedding_type",
        choices=["mean", "sum"],
        default="mean",
        help="Set embedding bag type",
    )
    parser.add_argument(
        "--use_sparse", type=str2bool, default=False, help="Use sparse in model"
    )
    parser.add_argument(
        "--use_article_emb",
        type=str2bool,
        default=False,
        help="Use article embeddings in model",
    )
    parser.add_argument(
        "--target_publication",
        type=int,
        default=0,
        help="Designate target publication.",
    )
