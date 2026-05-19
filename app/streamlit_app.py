import streamlit as st

from llm_helper import generate_llm_answer
from retriever import load_vector_store, retrieve_top_k


@st.cache_resource
def cached_vector_store():
    return load_vector_store()


st.set_page_config(
    page_title="Olasılık-İstatistik RAG Asistanı",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Olasılık-İstatistik RAG Asistanı")

st.write(
    "Bu uygulama, yapılandırılmış olasılık-istatistik veri setinden "
    "embedding tabanlı arama yapar ve bulunan kaynaklara göre LLM cevabı üretir."
)

# Sohbet geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []


# Vektör veritabanını yükle
try:
    embeddings, metadata = cached_vector_store()
    vector_store_ready = True
except FileNotFoundError:
    embeddings, metadata = None, None
    vector_store_ready = False


# Sidebar ayarları
st.sidebar.title("Ayarlar")

top_k = st.sidebar.slider(
    "Kaç kaynak getirilsin?",
    min_value=1,
    max_value=5,
    value=3
)

use_llm = st.sidebar.checkbox(
    "LLM ile cevap üret",
    value=True
)

response_style = st.sidebar.selectbox(
    "Cevap uzunluğu",
    ["Kısa", "Orta", "Ayrıntılı"],
    index=1
)

student_level = st.sidebar.selectbox(
    "Öğrenci seviyesi",
    ["Lise", "Lisans", "Yüksek Lisans"],
    index=1
)

show_sources = st.sidebar.checkbox(
    "Bulunan kaynakları göster",
    value=True
)

if vector_store_ready:
    st.sidebar.write("Toplam kayıt:", len(metadata))
    st.sidebar.write("Arama: OpenAI Embedding + Cosine Similarity")
else:
    st.error(
        "Embedding dosyaları bulunamadı. Önce şu komutu çalıştırın:\n\n"
        "`python app\\build_embeddings.py`"
    )


st.divider()

# Soru giriş alanı
st.subheader("Soru Sor")

question = st.text_input(
    "Sorunuzu yazın:",
    placeholder="Örneğin: Binom dağılımının varyansı nedir?",
    key="question_input"
)

col1, col2 = st.columns([1, 1])

with col1:
    ask_button = st.button("Cevapla", type="primary")

with col2:
    clear_button = st.button("Sohbeti Temizle")


if clear_button:
    st.session_state.messages = []
    st.rerun()


if ask_button:
    if not vector_store_ready:
        st.warning("Önce embedding dosyalarını oluşturmalısınız: `python app\\build_embeddings.py`")

    elif question.strip() == "":
        st.warning("Lütfen bir soru yazın.")

    else:
        # Kullanıcı mesajını geçmişe ekle
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        retrieved_items = retrieve_top_k(
            question=question,
            embeddings=embeddings,
            metadata=metadata,
            top_k=top_k
        )

        enriched_question = f"""
Kullanıcı sorusu:
{question}

Cevap ayarları:
- Öğrenci seviyesi: {student_level}
- Cevap uzunluğu: {response_style}

Lütfen cevabı bu seviyeye ve uzunluğa uygun hazırla.
"""

        if use_llm:
            with st.spinner("LLM cevabı hazırlanıyor..."):
                final_answer = generate_llm_answer(
                    enriched_question,
                    retrieved_items
                )
        else:
            final_answer = retrieved_items[0]["output"]

        # Asistan cevabını geçmişe ekle
        st.session_state.messages.append(
            {"role": "assistant", "content": final_answer}
        )

        st.success("Cevap hazır.")


# Sohbet geçmişi gösterimi
if st.session_state.messages:
    st.divider()
    st.subheader("Sohbet Geçmişi")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown("#### 👤 Soru")
            st.write(msg["content"])
        else:
            st.markdown("#### 🤖 Cevap")
            st.write(msg["content"])


# Son soruya ait kaynakları göster
if show_sources and "messages" in st.session_state and len(st.session_state.messages) > 0:
    if ask_button and question.strip() != "" and vector_store_ready:
        st.divider()
        st.subheader("Bulunan Kaynaklar")

        for i, item in enumerate(retrieved_items, start=1):
            with st.expander(f"Kaynak {i} - Skor: {item['score']:.3f}"):
                st.write("Konu:", item["topic"])
                st.write("Seviye:", item["level"])
                st.write("Cevap tipi:", item["type"])
                st.write("Eşleşen soru:", item["instruction"])
                st.write("Temel cevap:", item["output"])