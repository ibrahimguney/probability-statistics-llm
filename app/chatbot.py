import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "probability_statistics_qa.jsonl"


def load_dataset():
    data = []

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    return data


def simple_chatbot(question, data):
    question_lower = question.lower().strip()

    for item in data:
        if question_lower in item["instruction"].lower():
            return item["output"]

    return "Bu soruya henüz veri setimizde cevap yok."


if __name__ == "__main__":
    data = load_dataset()

    print("Olasılık-İstatistik Öğretici Chatbot")
    print("Çıkmak için 'q' yazın.\n")

    while True:
        question = input("Soru: ")

        if question.lower().strip() == "q":
            print("Görüşmek üzere.")
            break

        answer = simple_chatbot(question, data)
        print("Cevap:", answer)
        print()