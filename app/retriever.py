import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = BASE_DIR / "data" / "qa_embeddings.npy"
METADATA_PATH = BASE_DIR / "data" / "qa_metadata.jsonl"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_vector_store():
    embeddings = np.load(EMBEDDINGS_PATH)

    metadata = []

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            metadata.append(json.loads(line))

    return embeddings, metadata


def get_query_embedding(question):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
        encoding_format="float",
    )

    return np.array(response.data[0].embedding, dtype=np.float32)


def retrieve_top_k(question, embeddings, metadata, top_k=3):
    query_embedding = get_query_embedding(question).reshape(1, -1)

    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:
        item = metadata[index]
        item_with_score = dict(item)
        item_with_score["score"] = float(similarities[index])
        results.append(item_with_score)

    return results