from pathlib import Path
import json
import pickle

from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "rag_index"


def load_index():
    documents_path = INDEX_DIR / "documents.json"
    vectorizer_path = INDEX_DIR / "vectorizer.pkl"
    matrix_path = INDEX_DIR / "matrix.pkl"

    if not documents_path.exists():
        raise FileNotFoundError(
            "RAG indeksi bulunamadı. Önce proje ana klasöründe 'python ingest.py' çalıştırın."
        )

    with open(documents_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    with open(matrix_path, "rb") as f:
        matrix = pickle.load(f)

    return documents, vectorizer, matrix


def course_to_folder(course: str) -> str:
    course_map = {
        "Olasılık": "olasilik",
        "İstatistik": "istatistik",
        "SPSS": "spss",
        "Regresyon": "regresyon",
        "Araştırma Yöntemleri": "arastirma_yontemleri",
    }

    return course_map.get(course, "")


def retrieve_context(question: str, top_k: int = 3, course: str = "Genel"):
    documents, vectorizer, matrix = load_index()

    folder_filter = course_to_folder(course)

    candidate_indices = []

    for i, doc in enumerate(documents):
        source = doc.get("source", "")

        if course == "Genel":
            candidate_indices.append(i)
        elif source.startswith(folder_filter + "/") or source.startswith(folder_filter + "\\"):
            candidate_indices.append(i)

    if not candidate_indices:
        return []

    query_vector = vectorizer.transform([question])
    similarities = cosine_similarity(query_vector, matrix).flatten()

    ranked_indices = sorted(
        candidate_indices,
        key=lambda idx: similarities[idx],
        reverse=True
    )[:top_k]

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

    documents, vectorizer, matrix = load_index()

    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, matrix).flatten()

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


def clean_text(text: str) -> str:
    replacements = {
        "A B": "A ∩ B",
        "P(A B)": "P(A ∩ B)",
        "P(A B)/P(B)": "P(A ∩ B) / P(B)",
        "  ": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


def generate_simple_answer(question: str, contexts):
    if not contexts:
        return "Bu soruya cevap verebilmek için ilgili ders notu parçası bulunamadı."

    best = contexts[0]
    text = clean_text(best["text"])

    answer = f"""
### Soru
{question}

### Ders Notlarına Dayalı Cevap

{text}

### Kaynak Bilgisi

Bu cevap aşağıdaki ders notu parçasına dayanmaktadır:

- Kaynak: {best["source"]}
- Parça No: {best["chunk_id"]}
- Benzerlik Skoru: {best["score"]:.4f}
""".strip()

    return answer
    if not contexts:
        return "Bu soruya cevap verebilmek için ilgili ders notu parçası bulunamadı."

    best = contexts[0]

    answer = f"""
Soru: {question}

Ders notlarına göre en ilgili açıklama şudur:

{best["text"]}

Bu cevap, özellikle şu kaynaktan elde edilmiştir:
- Kaynak: {best["source"]}
- Parça No: {best["chunk_id"]}
- Benzerlik Skoru: {best["score"]:.4f}
""".strip()

    return answer


def answer_question(question: str, top_k: int = 3):
    contexts = retrieve_context(question, top_k=top_k)
    answer = generate_simple_answer(question, contexts)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }


if __name__ == "__main__":
    print("RAG Cevap Sistemi")
    print("Çıkmak için q yazın.")
    print("-" * 50)

    while True:
        question = input("\nSorunuz: ").strip()

        if question.lower() in ["q", "quit", "exit", "çıkış"]:
            print("Program kapatıldı.")
            break

        if not question:
            continue

        result = answer_question(question)

        print("\nCEVAP")
        print("=" * 50)
        print(result["answer"])

        print("\nKULLANILAN KAYNAK PARÇALAR")
        print("=" * 50)

        for i, ctx in enumerate(result["contexts"], start=1):
            print(f"\n{i}. Kaynak: {ctx['source']} | Parça: {ctx['chunk_id']} | Skor: {ctx['score']:.4f}")
            print(ctx["text"][:500])
            print("-" * 50)