import streamlit as st

from llm_helper import generate_llm_answer, generate_questions
from retriever import load_vector_store, retrieve_top_k


@st.cache_resource
def cached_vector_store():
    return load_vector_store()


st.set_page_config(
    page_title="Olasılık-İstatistik RAG Asistanı",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Olasılık-İstatistik Öğretici Platformu")

st.write(
    "Bu uygulama iki bölümden oluşur: RAG tabanlı konu asistanı ve "
    "olasılık-istatistik soru üretici modülü."
)


# Sohbet geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = ""


# Vektör veritabanını yükle
try:
    embeddings, metadata = cached_vector_store()
    vector_store_ready = True
except FileNotFoundError:
    embeddings, metadata = None, None
    vector_store_ready = False


# Sidebar
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
    st.sidebar.error("Embedding dosyaları bulunamadı.")


# Sekmeler
tab1, tab2 = st.tabs(
    [
        "🤖 RAG Asistanı",
        "📝 Soru Üretici"
    ]
)


# =========================================================
# 1. SEKME: RAG ASİSTANI
# =========================================================
with tab1:
    st.subheader("RAG Asistanı")

    st.write(
        "Bu bölümde soru sorabilirsiniz. Sistem önce veri setinde en yakın kaynakları bulur, "
        "sonra LLM ile açıklayıcı cevap üretir."
    )

    if not vector_store_ready:
        st.error(
            "Embedding dosyaları bulunamadı. Önce şu komutu çalıştırın:\n\n"
            "`python app\\build_embeddings.py`"
        )

    question = st.text_input(
        "Sorunuzu yazın:",
        placeholder="Örneğin: Binom dağılımının varyansı nedir?",
        key="rag_question_input"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        ask_button = st.button("Cevapla", type="primary", key="ask_rag_button")

    with col2:
        clear_button = st.button("Sohbeti Temizle", key="clear_chat_button")

    if clear_button:
        st.session_state.messages = []
        st.rerun()

    if ask_button:
        if not vector_store_ready:
            st.warning("Önce embedding dosyalarını oluşturmalısınız: `python app\\build_embeddings.py`")

        elif question.strip() == "":
            st.warning("Lütfen bir soru yazın.")

        else:
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

            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer}
            )

            st.success("Cevap hazır.")

            if show_sources:
                st.divider()
                st.subheader("Bulunan Kaynaklar")

                for i, item in enumerate(retrieved_items, start=1):
                    with st.expander(f"Kaynak {i} - Skor: {item['score']:.3f}"):
                        st.write("Konu:", item["topic"])
                        st.write("Seviye:", item["level"])
                        st.write("Cevap tipi:", item["type"])
                        st.write("Eşleşen soru:", item["instruction"])
                        st.write("Temel cevap:", item["output"])

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


# =========================================================
# 2. SEKME: SORU ÜRETİCİ
# =========================================================
with tab2:
    st.subheader("Soru Üretici")

    st.write(
        "Bu bölümde seçilen konuya ve seviyeye göre çözümlü soru, çoktan seçmeli soru "
        "veya kısa cevaplı soru üretebilirsiniz."
    )

    question_topics = [
        "Temel Olasılık",
        "Koşullu Olasılık",
        "Bayes Teoremi",
        "Sayma Yöntemleri",
        "Rassal Değişkenler",
        "Beklenen Değer ve Varyans",
        "Bernoulli Dağılımı",
        "Binom Dağılımı",
        "Poisson Dağılımı",
        "Normal Dağılım",
        "Güven Aralığı",
        "Hipotez Testleri",
        "t Testleri",
        "ANOVA",
        "Korelasyon",
        "Regresyon",
        "SPSS Yorumlama",
        "Python Uygulamaları"
    ]

    selected_topic = st.selectbox(
        "Konu seçin:",
        question_topics,
        index=7
    )

    selected_level = st.selectbox(
        "Seviye seçin:",
        ["Başlangıç", "Orta", "İleri"],
        index=1
    )

    selected_question_type = st.selectbox(
        "Soru türü seçin:",
        [
            "Çoktan Seçmeli",
            "Klasik",
            "Çözümlü Soru",
            "Kısa Cevaplı",
            "Doğru-Yanlış"
        ],
        index=0
    )

    question_count = st.slider(
        "Soru sayısı:",
        min_value=1,
        max_value=10,
        value=5
    )

    generate_button = st.button(
        "Soru Üret",
        type="primary",
        key="generate_questions_button"
    )

    if generate_button:
        with st.spinner("Sorular hazırlanıyor..."):
            questions_output = generate_questions(
                topic=selected_topic,
                level=selected_level,
                question_type=selected_question_type,
                question_count=question_count
            )

        st.session_state.generated_questions = questions_output
        st.success("Sorular hazır.")

    if st.session_state.generated_questions:
        st.divider()
        st.subheader("Üretilen Sorular")
        st.markdown(st.session_state.generated_questions)

        st.download_button(
            label="Soruları .txt olarak indir",
            data=st.session_state.generated_questions,
            file_name="uretilen_sorular.txt",
            mime="text/plain"
        )