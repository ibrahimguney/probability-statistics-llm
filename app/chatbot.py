import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "probability_statistics_qa.jsonl"


def load_dataset():
    data = []

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    return data


def build_search_engine(data):
    questions = [item["instruction"] for item in data]

    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)

    return vectorizer, question_vectors


def find_best_answer(question, data, vectorizer, question_vectors):
    user_vector = vectorizer.transform([question])
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < 0.20:
        return "Bu soruya henüz veri setimizde güçlü bir cevap yok.", best_score

    return data[best_index]["output"], best_score


if __name__ == "__main__":
    data = load_dataset()
    vectorizer, question_vectors = build_search_engine(data)

    print("Olasılık-İstatistik Öğretici Chatbot")
    print("Çıkmak için 'q' yazın.\n")

    while True:
        question = input("Soru: ")

        if question.lower().strip() == "q":
            print("Görüşmek üzere.")
            break

        answer, score = find_best_answer(
            question,
            data,
            vectorizer,
            question_vectors
        )

        print("Cevap:", answer)
        print("Benzerlik skoru:", round(score, 3))
        print()