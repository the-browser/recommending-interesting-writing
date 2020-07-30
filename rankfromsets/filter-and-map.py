import ujson as json
import argparse
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from data_processing.articles import Articles
import data_processing.dictionaries as dictionary
import os

# get arguments for script and parse
def expand_path(string):
    return Path(os.path.expandvars(string))


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Filter and Map Data")

parser.add_argument(
    "--dataset_path", type=expand_path, required=True, help="Path to Raw Dataset",
)

parser.add_argument(
    "--filter",
    type=str2bool,
    default=True,
    help="Whether to apply filtering to the raw dataset",
)

parser.add_argument(
    "--min_length", type=int, default=0, help="Minimum Length for Articles in Dataset",
)

parser.add_argument(
    "--days", type=int, help="How many days old articles can be.",
)

parser.add_argument(
    "--output_data_path", type=expand_path, help="File name to save data to."
)
parser.add_argument(
    "--dict_dir", type=expand_path, help="Path where dictionaries are located.",
)
parser.add_argument(
    "--tokenizer_file", type=str, help="Designate tokenizer source file.",
)

args = parser.parse_args()
tokenizer = BertWordPieceTokenizer(args.tokenizer_file, lowercase=True)

dictionary_dir = Path(args.dict_dir)
final_word_ids, final_url_ids, final_publication_ids = dictionary.load_dictionaries(
    dictionary_dir
)
print("Dictionaries loaded.")

if args.filter:
    raw_dataset = Articles(args.dataset_path)
    print("Initial: ", len(raw_dataset))
    if args.days is not None:
        filtered_data = raw_dataset.map_items(
            tokenizer,
            final_url_ids,
            final_publication_ids,
            filter=True,
            min_length=args.min_length,
            day_range=args.days,
        )
    else:
        filtered_data = raw_dataset.map_items(
            tokenizer,
            final_url_ids,
            final_publication_ids,
            filter=True,
            min_length=args.min_length,
        )

    # save mapped data for easier access next iteration
    data_path = Path(args.output_data_path)
    with open(data_path, "w") as file:
        json.dump(filtered_data, file)
    final_data = Articles(data_path)
    print("Final: ", len(final_data))
    print(f"Filtered, Mapped Data saved to {args.output_data_path}")
    print("-------------------")
else:
    final_data = Articles(args.dataset_path)
    print(f"Total Length: ", len(final_data))
    final_data.map_items(tokenizer, final_url_ids, final_publication_ids)

    # save mapped data
    data_path = Path(args.output_data_path)
    with open(data_path, "w") as file:
        json.dump(proper_data, file)
