from pathlib import Path
import json
import pickle

from sklearn.metrics.pairwise import cosine_similarity


INDEX_DIR = Path("data/rag_index")


def load_index():
    documents_path = INDEX_DIR / "documents.json"
    vectorizer_path = INDEX_DIR / "vectorizer.pkl"
    matrix_path = INDEX_DIR / "matrix.pkl"

    if not documents_path.exists():
        raise FileNotFoundError("documents.json bulunamadı. Önce python ingest.py çalıştırın.")

    if not vectorizer_path.exists():
        raise FileNotFoundError("vectorizer.pkl bulunamadı. Önce python ingest.py çalıştırın.")

    if not matrix_path.exists():
        raise FileNotFoundError("matrix.pkl bulunamadı. Önce python ingest.py çalıştırın.")

    with open(documents_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    with open(matrix_path, "rb") as f:
        matrix = pickle.load(f)

    return documents, vectorizer, matrix


def search(query: str, top_k: int = 3):
    documents, vectorizer, matrix = load_index()

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, matrix).flatten()

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for idx in ranked_indices:
        results.append(
            {
                "score": float(similarities[idx]),
                "source": documents[idx]["source"],
                "chunk_id": documents[idx]["chunk_id"],
                "text": documents[idx]["text"],
            }
        )

    return results


def main():
    print("RAG Arama Sistemi")
    print("Çıkmak için q yazın.")
    print("-" * 50)

    while True:
        query = input("\nSorunuz: ").strip()

        if query.lower() in ["q", "quit", "exit", "çıkış"]:
            print("Program kapatıldı.")
            break

        if not query:
            continue

        results = search(query, top_k=3)

        print("\nEn ilgili kaynak parçalar:")
        print("=" * 50)

        for i, result in enumerate(results, start=1):
            print(f"\n{i}. Sonuç")
            print(f"Kaynak: {result['source']}")
            print(f"Parça No: {result['chunk_id']}")
            print(f"Benzerlik Skoru: {result['score']:.4f}")
            print("-" * 50)
            print(result["text"])
            print("-" * 50)


if __name__ == "__main__":
    main()