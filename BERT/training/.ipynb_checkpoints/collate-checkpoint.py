# define function to return necessary data for dataloader to pass into model
import numpy as np
import torch

# add special_tokens and generate attention masks for an input
def prepare_input(ids):
    altered = ids
    altered.insert(0, 101)
    altered.append(102)
    zeroes_to_add = 512 - len(altered)
    if zeroes_to_add > 0:
        altered += [0] * zeroes_to_add
    altered = np.array(altered)
    altered_mask = np.where(altered != 0, 1, 0)
    return altered, altered_mask


# define method for sending data in each batch
def collate_fn(examples):
    words = []
    labels = []
    word_subsets = []
    masks = []
    for example in examples:
        tokens, mask = prepare_input(example["text"][:510])
        words.append(tokens)
        masks.append(mask)
        """
        for idx in range((len(example["text"]) // 510) + 1):
            current = example["text"][(idx * 510) : ((idx + 1) * 510)]
            fixed_tokens, current_mask = prepare_input(current)
            words.append(fixed_tokens)
            masks.append(current_mask)
        """
        word_subsets.append((len(example["text"]) // 510) + 1)
        labels.append(example["model_publication"])
    word_attributes = torch.tensor(words, dtype=torch.long)
    word_subsets.insert(0, 0)
    word_subset_counts = torch.tensor(np.cumsum(word_subsets), dtype=torch.long)
    real_labels = torch.tensor(labels, dtype=torch.long)
    attention_masks = torch.tensor(masks, dtype=torch.long)
    return (word_attributes, attention_masks, word_subset_counts, real_labels)
