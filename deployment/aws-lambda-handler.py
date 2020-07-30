import json
import numpy as np
import boto3
import scipy
import scipy.sparse
from io import BytesIO
import os
import base64

ACCESS_KEY = os.environ["ACCESS_KEY"]
SECRET_ACCESS_KEY = os.environ["SECRET_ACCESS_KEY"]


def getData(dataset):
    BUCKET = "personal-bucket-news-ranking"
    client = boto3.client(
        "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY
    )
    path_to_data = "sets/csr_articles_" + dataset + ".npz"
    result = client.get_object(Bucket=BUCKET, Key=path_to_data)
    word_articles = scipy.sparse.load_npz(BytesIO(result["Body"].read()))

    FILE_TO_READ = "word_emb.npy"
    result = client.get_object(Bucket=BUCKET, Key=FILE_TO_READ)
    word_emb = np.load(BytesIO(result["Body"].read()))

    FILE_TO_READ = "word_bias.npy"
    result = client.get_object(Bucket=BUCKET, Key=FILE_TO_READ)
    word_bias = np.load(BytesIO(result["Body"].read()))

    FILE_TO_READ = "reversed_word_ids.json"
    result = client.get_object(Bucket=BUCKET, Key=FILE_TO_READ)
    id_to_word = json.loads(result["Body"].read().decode())

    FILE_TO_READ = "sets/mapped_dataset_" + dataset + ".json"
    result = client.get_object(Bucket=BUCKET, Key=FILE_TO_READ)
    real_data = json.loads(result["Body"].read().decode())

    return word_articles, word_emb, word_bias, id_to_word, real_data


def lambda_handler(event, context):
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
    emb_dict = json.loads(event["body"])
    print(emb_dict)

    crime_scalar = 1 if emb_dict["crime"] == 0 else emb_dict["crime"]
    technology_scalar = 1 if emb_dict["technology"] == 0 else emb_dict["technology"]
    business_scalar = 1 if emb_dict["business"] == 0 else emb_dict["business"]
    politics_scalar = 1 if emb_dict["politics"] == 0 else emb_dict["politics"]
    word_articles, word_emb, word_bias, id_to_word, real_data = getData(
        emb_dict["dataset"]
    )

    technologyindices = [
        6786,
        6627,
        10660,
        2671,
        4087,
        2640,
        3941,
        3226,
        2470,
        4007,
        3330,
        2373,
        2291,
        2458,
        3274,
    ]

    crimeindices = [
        6997,
        4735,
        4028,
        12290,
        2610,
        2395,
        3827,
        6547,
        2375,
        2373,
        2331,
        2895,
        3036,
        4808,
        2111,
    ]

    politicsindices = [
        2576,
        3761,
        8801,
        10317,
        2231,
        3343,
        7072,
        3323,
        3864,
        3314,
        5543,
        2373,
        2602,
        3537,
        3821,
        3049,
        5656,
    ]

    businessindices = [
        2194,
        2147,
        5661,
        2166,
        2326,
        2034,
        3119,
        3293,
        3316,
        4411,
        6883,
        9926,
        6088,
        3171,
        3068,
    ]
    print("Data loaded successfully!")

    # update words related to specific topics based on scalar values provided
    weight = np.ones(len(word_emb))
    dot_product_sign = np.sign(word_emb.dot(publication_emb))
    if crime_scalar != 1:
        weight[crimeindices] *= crime_scalar * dot_product_sign[crimeindices]
    if politics_scalar != 1:
        weight[politicsindices] *= politics_scalar * dot_product_sign[politicsindices]
    if business_scalar != 1:
        weight[businessindices] *= business_scalar * dot_product_sign[businessindices]
    if technology_scalar != 1:
        weight[technologyindices] *= (
            technology_scalar * dot_product_sign[technologyindices]
        )
    word_emb = word_emb * weight.reshape(len(word_emb), 1)

    article_embeddings = word_articles.dot(word_emb)

    emb_times_publication = np.dot(
        article_embeddings, publication_emb.reshape(len(publication_emb), 1)
    )

    article_bias = word_articles.dot(word_bias)

    product_with_bias = emb_times_publication + article_bias

    word_counts = word_articles.sum(axis=1).reshape(word_articles.shape[0], 1)

    final_logits = np.divide(product_with_bias, word_counts) + float(publication_bias)

    indices = final_logits.argsort(axis=0)[-75:].reshape(75)

    word_logits = (
        np.dot(word_emb, publication_emb.reshape(len(publication_emb), 1)) + word_bias
    )

    top_articles = word_articles[indices.tolist()[0]]

    broadcasted_words_per_article = top_articles.toarray() * word_logits.T

    sorted_word_indices = broadcasted_words_per_article.argsort(axis=1)

    return_articles = []

    i = 0
    for idx in indices.tolist()[0]:
        current_article = real_data[int(idx)]
        current_article["logit"] = round(float(final_logits[int(idx)]), 3)
        current_sorted_words = sorted_word_indices[i]
        top_words = []
        least_words = []
        for top_word in current_sorted_words[-20:]:
            word = id_to_word[str(top_word)]
            if "##" not in word and "[UNK]" not in word:
                top_words.append(word)
        for least_word in current_sorted_words[:20]:
            word = id_to_word[str(least_word)]
            if "##" not in word and "[UNK]" not in word:
                least_words.append(word)
        current_article["top_words"] = top_words
        current_article["least_words"] = least_words
        return_articles.append(current_article)
        i += 1
    ordered_return_articles = return_articles[::-1]
    origin = event["headers"]["origin"]
    if origin in ["https://the-browser.github.io", "https://rohanbansal12.github.io"]:
        response = {
            "statusCode": 200,
            "body": json.dumps(ordered_return_articles),
            "headers": {"Access-Control-Allow-Origin": origin,},
        }
        return response
    else:
        response = {
            "statusCode": 500,
        }
        return response
