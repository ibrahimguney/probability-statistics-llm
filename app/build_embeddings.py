print("build_embeddings.py çalıştı")
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "probability_statistics_qa_structured.jsonl"
EMBEDDINGS_PATH = BASE_DIR / "data" / "qa_embeddings.npy"
METADATA_PATH = BASE_DIR / "data" / "qa_metadata.jsonl"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def make_embedding_text(item):
    return "\n".join(
        [
            f"Konu: {item.get('topic', '')}",
            f"Seviye: {item.get('level', '')}",
            f"Tür: {item.get('type', '')}",
            f"Soru: {item.get('instruction', '')}",
            f"Girdi: {item.get('input', '')}",
            f"Cevap: {item.get('output', '')}",
        ]
    )


def load_dataset():
    data = []

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    return data


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float",
    )

    return response.data[0].embedding


def main():
    data = load_dataset()

    embeddings = []

    print(f"Toplam kayıt: {len(data)}")

    with open(METADATA_PATH, "w", encoding="utf-8") as metadata_file:
        for index, item in enumerate(data, start=1):
            text = make_embedding_text(item)

            print(f"Embedding oluşturuluyor: {index}/{len(data)} - {item['instruction']}")

            embedding = get_embedding(text)
            embeddings.append(embedding)

            metadata_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.save(EMBEDDINGS_PATH, embeddings_array)

    print("Embedding dosyası kaydedildi:", EMBEDDINGS_PATH)
    print("Metadata dosyası kaydedildi:", METADATA_PATH)
    print("Embedding boyutu:", embeddings_array.shape)


if __name__ == "__main__":
    main()