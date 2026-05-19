import json
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "probability_statistics_qa.jsonl"


@st.cache_data
def load_dataset():
    data = []

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    return data


@st.cache_resource
def build_search_engine(questions):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)

    return vectorizer, question_vectors


def find_best_answer(question, data, vectorizer, question_vectors):
    user_vector = vectorizer.transform([question])
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < 0.20:
        return (
            "Bu soruya henüz veri setimizde güçlü bir cevap yok. "
            "Bu konuyu veri setine ekleyebiliriz.",
            best_score,
            None
        )

    return data[best_index]["output"], best_score, data[best_index]["instruction"]


st.set_page_config(
    page_title="Olasılık-İstatistik Öğretici Asistan",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Olasılık-İstatistik Öğretici Asistan")

st.write(
    "Bu uygulama, olasılık ve istatistik konularında hazırlanmış "
    "küçük bir soru-cevap veri seti üzerinden çalışır."
)

data = load_dataset()
questions = [item["instruction"] for item in data]
vectorizer, question_vectors = build_search_engine(questions)

st.sidebar.title("Proje Bilgisi")
st.sidebar.write("Veri setindeki örnek sayısı:", len(data))
st.sidebar.write("Yöntem: TF-IDF + Cosine Similarity")
st.sidebar.write("Durum: İlk prototip")

question = st.text_input(
    "Sorunuzu yazın:",
    placeholder="Örneğin: Koşullu olasılık nedir?"
)

if st.button("Cevapla"):
    if question.strip() == "":
        st.warning("Lütfen bir soru yazın.")
    else:
        answer, score, matched_question = find_best_answer(
            question,
            data,
            vectorizer,
            question_vectors
        )

        st.subheader("Cevap")
        st.write(answer)

        st.subheader("Benzerlik Bilgisi")
        st.write("Benzerlik skoru:", round(score, 3))

        if matched_question is not None:
            st.write("Eşleşen soru:", matched_question)

st.divider()

st.subheader("Örnek Sorular")

example_questions = [
    "Olasılık nedir?",
    "Koşullu olasılık nedir?",
    "Beklenen değer nedir?",
    "Örnek uzay nedir?",
    "Olay nedir?"
]

for ex in example_questions:
    st.code(ex)