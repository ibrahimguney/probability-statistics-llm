import json
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_helper import generate_llm_answer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "probability_statistics_qa_structured.jsonl"


@st.cache_data
def load_dataset():
    data = []

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    return data


@st.cache_resource
def build_search_engine(search_texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(search_texts)
    return vectorizer, vectors


def make_search_text(item):
    return " ".join(
        [
            item.get("topic", ""),
            item.get("level", ""),
            item.get("type", ""),
            item.get("instruction", ""),
            item.get("input", ""),
            item.get("output", "")
        ]
    )


def find_best_answer(question, data, vectorizer, vectors):
    user_vector = vectorizer.transform([question])
    similarities = cosine_similarity(user_vector, vectors)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]
    best_item = data[best_index]

    if best_score < 0.20:
        return None, best_score

    return best_item, best_score


st.set_page_config(
    page_title="Olasılık-İstatistik LLM Asistanı",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Olasılık-İstatistik LLM Asistanı")

st.write(
    "Bu uygulama önce kendi yapılandırılmış veri setimizde en yakın bilgiyi bulur, "
    "sonra LLM kullanarak daha doğal ve öğretici bir cevap üretir."
)

data = load_dataset()

topics = sorted(set(item["topic"] for item in data))
levels = sorted(set(item["level"] for item in data))
types = sorted(set(item["type"] for item in data))

st.sidebar.title("Filtreler")

selected_topic = st.sidebar.selectbox(
    "Konu",
    ["Tümü"] + topics
)

selected_level = st.sidebar.selectbox(
    "Seviye",
    ["Tümü"] + levels
)

selected_type = st.sidebar.selectbox(
    "Cevap tipi",
    ["Tümü"] + types
)

use_llm = st.sidebar.checkbox(
    "LLM ile cevabı zenginleştir",
    value=True
)

filtered_data = data

if selected_topic != "Tümü":
    filtered_data = [item for item in filtered_data if item["topic"] == selected_topic]

if selected_level != "Tümü":
    filtered_data = [item for item in filtered_data if item["level"] == selected_level]

if selected_type != "Tümü":
    filtered_data = [item for item in filtered_data if item["type"] == selected_type]

st.sidebar.write("Toplam kayıt:", len(data))
st.sidebar.write("Filtrelenmiş kayıt:", len(filtered_data))
st.sidebar.write("Arama: TF-IDF + Cosine Similarity")
st.sidebar.write("Cevap üretimi: OpenAI Responses API")

question = st.text_input(
    "Sorunuzu yazın:",
    placeholder="Örneğin: Binom dağılımının varyansı nedir?"
)

if st.button("Cevapla"):
    if question.strip() == "":
        st.warning("Lütfen bir soru yazın.")
    elif len(filtered_data) == 0:
        st.warning("Seçilen filtrelerde kayıt yok.")
    else:
        filtered_search_texts = [make_search_text(item) for item in filtered_data]
        filtered_vectorizer, filtered_vectors = build_search_engine(filtered_search_texts)

        best_item, score = find_best_answer(
            question,
            filtered_data,
            filtered_vectorizer,
            filtered_vectors
        )

        if best_item is None:
            st.warning(
                "Bu soruya henüz veri setimizde güçlü bir cevap yok. "
                "Bu konuyu yeni veri olarak ekleyebiliriz."
            )
            st.write("Benzerlik skoru:", round(score, 3))
        else:
            if use_llm:
                with st.spinner("LLM cevabı hazırlanıyor..."):
                    final_answer = generate_llm_answer(question, best_item)
            else:
                final_answer = best_item["output"]

            st.subheader("Cevap")
            st.write(final_answer)

            st.subheader("Kaynak Veri Seti Kaydı")
            st.write("Konu:", best_item["topic"])
            st.write("Seviye:", best_item["level"])
            st.write("Cevap tipi:", best_item["type"])
            st.write("Eşleşen soru:", best_item["instruction"])
            st.write("Benzerlik skoru:", round(score, 3))

st.divider()

st.subheader("Veri Setinden Örnek Sorular")

for item in filtered_data[:10]:
    st.code(item["instruction"])