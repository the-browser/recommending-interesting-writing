import numpy as np
from tokenizers import BertWordPieceTokenizer
import torch
import ujson as json
import collections

# define Articles dataset class for easy sampling, iteration, and weight creating
class Articles(torch.utils.data.Dataset):
    def __init__(self, json_file, index_file=None):
        super().__init__()
        with open(json_file, "r") as data_file:
            self.examples = json.loads(data_file.read())

        if index_file is not None:
            with open(index_file, "r") as file:
                indices = [int(index.rstrip()) for index in file.readlines()]
            self.examples = [self.examples[i] for i in indices]

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def create_positive_sampler(self, target_publication):
        prob = np.zeros(len(self))
        for idx, example in enumerate(self.examples):
            if example["model_publication"] == target_publication:
                prob[idx] = 1
        return torch.utils.data.WeightedRandomSampler(
            weights=prob, num_samples=len(self), replacement=True
        )

    def create_negative_sampler(self, target_publication):
        prob = np.zeros(len(self))
        for idx, example in enumerate(self.examples):
            if example["model_publication"] != target_publication:
                prob[idx] = 1
        return torch.utils.data.WeightedRandomSampler(
            weights=prob, num_samples=len(self), replacement=True
        )

    def map_items(
        self,
        tokenizer,
        url_to_id,
        publication_to_id,
        filter=False,
        min_length=0,
        day_range=None,
    ):
        min_length_articles = []
        for idx, example in enumerate(self.examples):
            encoded = tokenizer.encode(example["text"], add_special_tokens=False).ids
            self.examples[idx]["text"] = encoded
            self.examples[idx]["url"] = url_to_id.get(
                example["url"], url_to_id.get("miscellaneous")
            )
            self.examples[idx]["model_publication"] = publication_to_id.get(
                example["model_publication"], publication_to_id.get("miscellaneous")
            )
            if filter:
                if day_range is not None:
                    dated = datetime.strptime(example["date"], "%Y-%m-%d")
                    now = datetime.now()
                    last_month = now - timedelta(days=day_range)
                    if (
                        len(self.examples[idx]["text"]) > min_length
                        and last_month <= dated <= now
                    ):
                        min_length_articles.append(self.examples[idx])
                else:
                    if len(self.examples[idx]["text"]) > min_length:
                        min_length_articles.append(self.examples[idx])
        return min_length_articles
