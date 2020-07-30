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
        "--train_path", type=expand_path, required=True, help="Path to train set"
    )
    parser.add_argument(
        "--test_path", type=expand_path, required=True, help="Path to test set"
    )
    parser.add_argument(
        "--eval_path", type=expand_path, required=True, help="Path to evaluation set"
    )
    parser.add_argument(
        "--data_dir",
        type=expand_path,
        required=False,
        help="Path for data storage and loading. Only required if dictionaries and mapped-data being created.",
    )
    parser.add_argument(
        "--output_dir",
        type=expand_path,
        required=True,
        help="Directory for storing results",
    )
    parser.add_argument(
        "--create_dicts",
        type=str2bool,
        default=True,
        help="Create dictionaries for articles, publications, and words.",
    )
    parser.add_argument(
        "--dict_dir",
        type=expand_path,
        help="Path where dictionaries are located. Required if --create_dict is False.",
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
        "--target_publication",
        type=int,
        default=0,
        help="Designate target publication.",
    )
    parser.add_argument(
        "--tokenizer_file", type=str, help="Designate tokenizer source file.",
    )
    parser.add_argument(
        "--index_file_path",
        type=expand_path,
        required=True,
        help="Designate randomized indices for Evaluation performance collection.",
    )


def add_training(parser):
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=500, help="Batch size for Evaluation"
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Use GPU for training and evaluation",
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=10000,
        help="Total Number Of Steps For Training",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Total Number Of Steps For Linear Learning Rate Warmup",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=1000,
        help="How often to calculate recall and test performance.",
    )
    parser.add_argument(
        "--eval_recall_max",
        type=int,
        default=100,
        help="What to calculate recall out of.",
    )
    parser.add_argument(
        "--test_recall_max",
        type=int,
        default=1000,
        help="What to calculate recall out of.",
    )


def add_optimization(parser):
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Set optimizer learning rate"
    )
    parser.add_argument(
        "--clip_grad",
        type=str2bool,
        default=True,
        help="Turn on gradient clipping to prevent exploding gradient problem",
    )


def add_model(parser):
    parser.add_argument(
        "--model_path", type=str, help="Path to pretrained model weights."
    )
