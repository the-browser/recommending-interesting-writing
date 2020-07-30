import csv
from tqdm import tqdm
import sys
from transformers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(
    "/scratch/gpfs/altosaar/dat/longform-data/BERT/bert-base-uncased.txt",
    lowercase=True,
)
csv.field_size_limit(sys.maxsize)
ifile = open("fake-news.csv", "r")
reader = csv.reader(ifile)
ofile = open("mapped-fake-news.csv", "w")
writer = csv.writer(ofile, delimiter=",")
header_row = ["title", "text", "url", "publication", "model_publication", "link"]
writer.writerow(header_row)
i = 0
for row in tqdm(reader):
    if len(row) != 17 or i == 0:
        i += 1
        continue
    current_row = []
    current_row.append(row[9])
    raw_text = row[5]
    id_text = tokenizer.encode(raw_text).ids
    id_text.pop()
    id_text.pop(0)
    current_row.append(id_text)
    current_row.append(8000000)
    current_row.append("fake-news-corpus")
    current_row.append(25)
    current_row.append(row[4])
    writer.writerow(current_row)
