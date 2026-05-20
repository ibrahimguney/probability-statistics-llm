import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# -------------------------------------------------
# TEMEL AYARLAR
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "docs" / "raw"
INDEX_DIR = BASE_DIR / "data" / "rag_index"

COURSE_FOLDERS = {
    "Olasılık": "olasilik",
    "İstatistik": "istatistik",
    "SPSS": "spss",
    "Regresyon": "regresyon",
    "Araştırma Yöntemleri": "arastirma_yontemleri",
}

COURSE_LIST = [
    "Genel",
    "Olasılık",
    "İstatistik",
    "SPSS",
    "Regresyon",
    "Araştırma Yöntemleri",
]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(name, default)


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

# -------------------------------------------------
# PROJE MODÜLLERİ
# -------------------------------------------------

from ingest import rebuild_index

try:
    from app.rag_answer import retrieve_context
except ModuleNotFoundError:
    from rag_answer import retrieve_context

try:
    from app.llm_rag_answer import answer_with_llm
except ModuleNotFoundError:
    from llm_rag_answer import answer_with_llm


# -------------------------------------------------
# STREAMLIT SAYFA AYARI
# -------------------------------------------------

st.set_page_config(
    page_title="Olasılık ve İstatistik LLM",
    page_icon="📘",
    layout="wide",
)


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False


# -------------------------------------------------
# YARDIMCI FONKSİYONLAR
# -------------------------------------------------
def show_admin_help():
    with st.expander("ℹ️ Yönetici Paneli Nasıl Kullanılır?", expanded=False):
        st.markdown(
            """
            **Yönetici Paneli**, ders notlarını sisteme eklemek ve RAG indeksini yenilemek için kullanılır.

            **Kullanım sırası:**

            1. Yönetici şifresiyle giriş yapın.
            2. Ders seçin.
            3. PDF veya DOCX ders notunu yükleyin.
            4. **Dosyayı Kaydet** butonuna basın.
            5. **RAG İndeksini Yenile** butonuna basın.

            Yeni dosya yükledikten veya dosya sildikten sonra mutlaka **RAG indeksini yenilemeniz** gerekir.
            """
        )


def show_rag_help():
    with st.expander("ℹ️ RAG Arama Nedir?", expanded=False):
        st.markdown(
            """
            **RAG Arama**, ders notları içinden sorunuzla en ilgili metin parçalarını bulur.

            Bu sayfa henüz LLM cevabı üretmez. Sadece şu bilgileri gösterir:

            - Kaynak dosya adı
            - Parça numarası
            - Benzerlik skoru
            - İlgili ders notu metni

            **Ne zaman kullanılır?**

            - Yüklenen dosyaların doğru indekslenip indekslenmediğini kontrol etmek için
            - Sorunun hangi kaynak parçalara dayandığını görmek için
            - LLM cevabından önce kaynakları test etmek için
            """
        )


def show_llm_help():
    with st.expander("ℹ️ LLM Destekli Cevap Nasıl Çalışır?", expanded=False):
        st.markdown(
            """
            **LLM Destekli Cevap** sayfası iki aşamalı çalışır:

            1. Önce seçilen derse ait ders notlarından ilgili kaynak parçalar bulunur.
            2. Sonra OpenAI modeli bu kaynaklara dayanarak açıklayıcı cevap üretir.

            **Önemli:**  
            Cevaplar, yüklenen ders notları temel alınarak üretilir. Ders notunda bilgi yoksa sistem bunu belirtmelidir.

            **İyi soru örnekleri:**

            - Koşullu olasılık nedir?
            - Bayes teoremi nasıl yorumlanır?
            - Standart sapma neyi gösterir?
            - Regresyonda R-kare nasıl açıklanır?
            """
        )


def show_student_help():
    with st.expander("ℹ️ Öğrenci İçin Soru Sorma Önerileri", expanded=False):
        st.markdown(
            """
            Daha iyi cevap almak için sorularınızı açık ve ders bağlamına uygun yazın.

            **İyi örnekler:**

            - Koşullu olasılığı formülle açıklar mısın?
            - Binom dağılımı hangi durumlarda kullanılır?
            - Standart sapma ile varyans arasındaki fark nedir?
            - SPSS'te ANOVA çıktısı nasıl yorumlanır?

            **Daha zayıf örnekler:**

            - Bunu anlat
            - Bu ne?
            - Örnek ver

            Kısa ama açık soru yazmak, daha doğru cevap alınmasını sağlar.
            """
        )
def create_course_folders():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for folder in COURSE_FOLDERS.values():
        (RAW_DIR / folder).mkdir(parents=True, exist_ok=True)


def list_course_files():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    return sorted(
        [
            file
            for file in RAW_DIR.rglob("*")
            if file.suffix.lower() in [".pdf", ".docx"]
        ]
    )


# -------------------------------------------------
# SOL MENÜ
# -------------------------------------------------

st.sidebar.title("📘 Menü")

page = st.sidebar.radio(
    "Sayfa seçiniz:",
    [
        "Ana Sayfa",
        "Yönetici Paneli",
        "RAG Arama",
        "LLM Destekli Cevap",
        "Dersler",
        "Proje Hakkında",
    ],
)

st.sidebar.divider()

st.sidebar.info(
    "Bu uygulama PDF ve Word ders notları üzerinden çalışan "
    "RAG tabanlı akademik asistan örneğidir."
)

st.sidebar.markdown("### Durum")

if (INDEX_DIR / "documents.json").exists():
    st.sidebar.success("RAG aktif")
else:
    st.sidebar.warning("RAG indeksi yok")

if get_secret("OPENAI_API_KEY"):
    st.sidebar.success("LLM aktif")
else:
    st.sidebar.warning("OPENAI_API_KEY yok")

st.sidebar.info("Ders notları: docs/raw")


# -------------------------------------------------
# ANA SAYFA
# -------------------------------------------------

if page == "Ana Sayfa":
    st.title("📘 Olasılık ve İstatistik LLM")

    st.write(
        "Bu uygulama, olasılık ve istatistik ders notları üzerinden çalışan "
        "RAG tabanlı akademik asistan örneğidir."
    )

    st.info(
        "Ders notları `docs/raw` klasörüne eklenir. "
        "Ardından Yönetici Paneli üzerinden RAG indeksi yenilenir."
    )

    st.markdown("## Uygulamada Neler Var?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📄 Ders Notları")
        st.write("PDF ve Word ders notları okunur.")

    with col2:
        st.subheader("🔎 RAG Arama")
        st.write("Soruya en uygun kaynak parçalar bulunur.")

    with col3:
        st.subheader("🤖 LLM Cevabı")
        st.write("OpenAI destekli açıklayıcı akademik cevap üretilir.")

    st.markdown("## Kısa Kullanım Kılavuzu")

    with st.expander("🚀 Uygulamayı İlk Kez Kullanıyorum"):
        st.markdown(
            """
            1. Önce **Yönetici Paneli**ne girin.
            2. Yönetici şifresini yazın.
            3. Ders seçerek PDF veya DOCX dosyası yükleyin.
            4. **RAG İndeksini Yenile** butonuna basın.
            5. **RAG Arama** sayfasında kaynakları test edin.
            6. **LLM Destekli Cevap** sayfasında soru sorun.
            """
        )

    with st.expander("📌 RAG İndeksi Ne Zaman Yenilenmeli?"):
        st.markdown(
            """
            Aşağıdaki durumlarda RAG indeksi yenilenmelidir:

            - Yeni ders notu yüklendiğinde
            - Ders notu silindiğinde
            - Ders notu değiştirildiğinde
            - Kaynak parçalar beklenen şekilde gelmediğinde
            """
        )

# -------------------------------------------------
# YÖNETİCİ PANELİ
# -------------------------------------------------

elif page == "Yönetici Paneli":
    st.title("🛠️ Yönetici Paneli")
    show_admin_help()

    if not st.session_state.admin_logged_in:
        st.warning("Bu sayfaya erişmek için yönetici şifresi gereklidir.")

        admin_password_input = st.text_input(
            "Yönetici şifresi:",
            type="password",
        )

        if st.button("Giriş Yap"):
            correct_password = get_secret("ADMIN_PASSWORD", "")

            if admin_password_input == correct_password and correct_password:
                st.session_state.admin_logged_in = True
                st.success("Yönetici girişi başarılı.")
                st.rerun()
            else:
                st.error("Şifre hatalı veya ADMIN_PASSWORD tanımlı değil.")

        st.stop()

    if st.button("Yönetici Çıkışı Yap"):
        st.session_state.admin_logged_in = False
        st.success("Yönetici oturumu kapatıldı.")
        st.rerun()

    st.write(
        "Bu bölümden ders notlarını yönetebilir, yeni PDF/DOCX dosyaları yükleyebilir, "
        "gerektiğinde dosya silebilir ve RAG indeksini yenileyebilirsiniz."
    )

    create_course_folders()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    st.divider()

    st.subheader("📌 Sistem Durumu")

    documents_json = INDEX_DIR / "documents.json"
    vectorizer_pkl = INDEX_DIR / "vectorizer.pkl"
    matrix_pkl = INDEX_DIR / "matrix.pkl"

    col1, col2, col3 = st.columns(3)

    with col1:
        if documents_json.exists():
            st.success("documents.json var")
        else:
            st.error("documents.json yok")

    with col2:
        if vectorizer_pkl.exists():
            st.success("vectorizer.pkl var")
        else:
            st.error("vectorizer.pkl yok")

    with col3:
        if matrix_pkl.exists():
            st.success("matrix.pkl var")
        else:
            st.error("matrix.pkl yok")

    st.divider()

    st.subheader("📤 Yeni Ders Notu Yükle")

    upload_course = st.selectbox(
        "Dosyanın ait olduğu dersi seçiniz:",
        [
            "Olasılık",
            "İstatistik",
            "SPSS",
            "Regresyon",
            "Araştırma Yöntemleri",
        ],
    )

    uploaded_file = st.file_uploader(
        "PDF veya DOCX dosyası seçiniz:",
        type=["pdf", "docx"],
    )

    if uploaded_file is not None:
        course_folder = COURSE_FOLDERS[upload_course]
        target_dir = RAW_DIR / course_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        save_path = target_dir / uploaded_file.name

        if save_path.exists():
            st.warning(
                f"`{uploaded_file.name}` adlı dosya zaten var. "
                "Kaydederseniz mevcut dosyanın üzerine yazılır."
            )

        if st.button("Dosyayı Kaydet"):
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Dosya kaydedildi: {uploaded_file.name}")
            st.code(str(save_path), language="text")

    st.divider()

    st.subheader("📚 Mevcut Ders Notları")

    files = list_course_files()

    if not files:
        st.warning("Henüz `docs/raw` klasöründe PDF veya DOCX dosyası yok.")
    else:
        for file in files:
            file_size_kb = file.stat().st_size / 1024

            col_file, col_size, col_delete = st.columns([5, 2, 2])

            with col_file:
                st.write(f"📄 {file.relative_to(RAW_DIR)}")

            with col_size:
                st.write(f"{file_size_kb:.1f} KB")

            with col_delete:
                if st.button("Sil", key=f"delete_{file.relative_to(RAW_DIR)}"):
                    try:
                        file.unlink()
                        st.success(f"Silindi: {file.name}")
                        st.rerun()
                    except Exception as e:
                        st.error("Dosya silinirken hata oluştu.")
                        st.exception(e)

    st.divider()

    st.subheader("🔄 RAG İndeksini Yenile")

    st.write(
        "Yeni dosya yükledikten veya dosya sildikten sonra "
        "RAG indeksini yeniden oluşturmalısınız."
    )

    if st.button("RAG İndeksini Yenile"):
        try:
            with st.spinner("Ders notları yeniden indeksleniyor..."):
                rebuild_index()

            st.success("RAG indeksi başarıyla yenilendi.")

        except Exception as e:
            st.error("İndeks yenilenirken hata oluştu.")
            st.exception(e)

    st.divider()

    st.subheader("📁 Klasör Bilgileri")

    st.code(
        f"""
Proje ana klasörü:
{BASE_DIR}

Ders notları klasörü:
{RAW_DIR}

RAG indeks klasörü:
{INDEX_DIR}
        """,
        language="text",
    )

# -------------------------------------------------
# RAG ARAMA
# -------------------------------------------------

elif page == "RAG Arama":
    st.title("🔎 RAG Arama")
    show_rag_help()

    st.write(
        "Bu bölüm sadece ders notlarından en ilgili kaynak parçaları getirir. "
        "Henüz LLM ile açıklayıcı cevap üretmez."
    )

    course = st.selectbox(
        "Ders seçiniz:",
        COURSE_LIST,
    )

    question = st.text_input(
        "Aranacak soru veya konu:",
        placeholder="Örneğin: Koşullu olasılık nedir?",
    )

    top_k = st.slider(
        "Kaç kaynak parça getirilsin?",
        min_value=1,
        max_value=5,
        value=3,
    )

    if st.button("Kaynak Parçaları Getir"):
        if not question.strip():
            st.warning("Lütfen bir soru veya konu yazınız.")
        else:
            try:
                with st.spinner("RAG indeksi aranıyor..."):
                    contexts = retrieve_context(
                        question,
                        top_k=top_k,
                        course=course,
                    )

                st.subheader("Bulunan Kaynak Parçalar")

                if not contexts:
                    st.warning(
                        "Seçilen derse ait indekslenmiş kaynak bulunamadı. "
                        "Yönetici Paneli’nden ilgili derse PDF/DOCX dosyası yükleyip "
                        "RAG indeksini yenileyin."
                    )
                else:
                    for i, ctx in enumerate(contexts, start=1):
                        with st.expander(
                            f"{i}. Kaynak: {ctx['source']} | "
                            f"Parça: {ctx['chunk_id']} | "
                            f"Skor: {ctx['score']:.4f}"
                        ):
                            st.write(ctx["text"])

            except FileNotFoundError as e:
                st.error(str(e))
                st.code("python ingest.py", language="bat")

            except Exception as e:
                st.error("Beklenmeyen bir hata oluştu.")
                st.exception(e)

# -------------------------------------------------
# LLM DESTEKLİ CEVAP
# -------------------------------------------------

elif page == "LLM Destekli Cevap":
    st.title("🤖 LLM Destekli RAG Cevap")
    show_llm_help()
    show_student_help()

    st.write(
        "Bu bölüm önce seçilen ders bağlamında ders notlarından ilgili parçaları bulur, "
        "sonra OpenAI modeliyle açıklayıcı akademik cevap üretir."
    )

    if st.button("Sohbet Geçmişini Temizle"):
        st.session_state.chat_history = []
        st.success("Sohbet geçmişi temizlendi.")

    st.divider()

    course = st.selectbox(
        "Ders seçiniz:",
        COURSE_LIST,
    )

    question = st.text_input(
        "Sorunuzu yazınız:",
        placeholder="Örneğin: Koşullu olasılık nedir?",
    )

    top_k = st.slider(
        "Kaç kaynak parça kullanılsın?",
        min_value=1,
        max_value=5,
        value=3,
    )

    if st.button("LLM ile Cevapla"):
        if not question.strip():
            st.warning("Lütfen bir soru yazınız.")
        else:
            try:
                with st.spinner("Ders notları taranıyor ve LLM cevabı hazırlanıyor..."):
                    result = answer_with_llm(
                        question,
                        top_k=top_k,
                        course=course,
                    )

                st.session_state.chat_history.append(
                    {
                        "course": course,
                        "question": question,
                        "answer": result["answer"],
                        "contexts": result["contexts"],
                    }
                )

            except FileNotFoundError as e:
                st.error(str(e))
                st.code("python ingest.py", language="bat")

            except Exception as e:
                st.error("Beklenmeyen bir hata oluştu.")
                st.exception(e)

    st.subheader("Sohbet Geçmişi")

    if not st.session_state.chat_history:
        st.info("Henüz soru sorulmadı.")
    else:
        for i, item in enumerate(reversed(st.session_state.chat_history), start=1):
            soru_no = len(st.session_state.chat_history) - i + 1

            st.markdown(f"### Soru {soru_no}")
            st.markdown(f"**Ders:** {item.get('course', 'Genel')}")
            st.markdown(f"**Soru:** {item['question']}")

            st.markdown("**Cevap:**")
            st.markdown(item["answer"])

            with st.expander("Kullanılan Kaynak Parçalar"):
                for j, ctx in enumerate(item["contexts"], start=1):
                    st.markdown(
                        f"**{j}. Kaynak:** {ctx['source']} | "
                        f"**Parça:** {ctx['chunk_id']} | "
                        f"**Skor:** {ctx['score']:.4f}"
                    )
                    st.write(ctx["text"])
                    st.divider()


# -------------------------------------------------
# DERSLER
# -------------------------------------------------

elif page == "Dersler":
    st.title("📚 Dersler")

    selected_course = st.selectbox(
        "Ders içeriği seçiniz:",
        [
            "Olasılık",
            "İstatistik",
            "SPSS",
            "Regresyon",
            "Araştırma Yöntemleri",
        ],
    )

    if selected_course == "Olasılık":
        st.markdown("### Olasılık")
        st.markdown(
            """
            - Deney, örnek uzay ve olay
            - Olasılık aksiyomları
            - Koşullu olasılık
            - Bağımsızlık
            - Bayes teoremi
            - Permütasyon ve kombinasyon
            - Rassal değişkenler
            - Binom, Poisson ve Normal dağılımlar
            - Merkezi Limit Teoremi
            """
        )

    elif selected_course == "İstatistik":
        st.markdown("### İstatistik")
        st.markdown(
            """
            - Betimsel istatistik
            - Ortalama, medyan ve mod
            - Varyans ve standart sapma
            - Grafikler ve tablolar
            - Örnekleme
            - Güven aralığı
            - Hipotez testi
            - Korelasyon ve regresyona giriş
            """
        )

    elif selected_course == "SPSS":
        st.markdown("### SPSS")
        st.markdown(
            """
            - Veri girişi
            - Değişken tanımlama
            - Betimsel istatistikler
            - Grafik oluşturma
            - t testi
            - ANOVA
            - Korelasyon
            - Regresyon
            - ANCOVA
            """
        )

    elif selected_course == "Regresyon":
        st.markdown("### Regresyon")
        st.markdown(
            """
            - Basit doğrusal regresyon
            - Çoklu regresyon
            - Katsayıların yorumu
            - Model anlamlılığı
            - R-kare ve düzeltilmiş R-kare
            - Varsayımlar
            - Artık analizi
            - Panel regresyona giriş
            """
        )

    elif selected_course == "Araştırma Yöntemleri":
        st.markdown("### Araştırma Yöntemleri")
        st.markdown(
            """
            - Araştırma problemi
            - Hipotez kurma
            - Evren ve örneklem
            - Veri toplama
            - Ölçek geliştirme
            - Geçerlik ve güvenirlik
            - Nicel araştırma desenleri
            - Raporlama
            """
        )


# -------------------------------------------------
# PROJE HAKKINDA
# -------------------------------------------------

elif page == "Proje Hakkında":
    st.title("ℹ️ Proje Hakkında")
    st.markdown("### Sayfaların Görevleri")

    st.markdown(
        """
        - **Ana Sayfa:** Uygulamanın genel tanıtımı ve kullanım sırası
        - **Yönetici Paneli:** Dosya yükleme, silme ve indeks yenileme
        - **RAG Arama:** Kaynak parçaları test etme
        - **LLM Destekli Cevap:** Ders notlarına dayalı akademik cevap üretme
        - **Dersler:** Ders başlıklarını görüntüleme
        - **Proje Hakkında:** Teknik yapı ve iş akışı
        """
    )

    st.write(
        "Bu proje, olasılık ve istatistik dersleri için hazırlanmış "
        "küçük ölçekli bir akademik yapay zekâ asistanıdır."
    )

    st.markdown("### Kullanılan Yapı")

    st.code(
        """
docs/raw/
    olasilik/
    istatistik/
    spss/
    regresyon/
    arastirma_yontemleri/

data/rag_index/
    documents.json
    vectorizer.pkl
    matrix.pkl

app/
    streamlit_app.py
    rag_answer.py
    llm_rag_answer.py

ingest.py
requirements.txt
.env
        """,
        language="text",
    )

    st.markdown("### İş Akışı")

    st.markdown(
        """
        1. Ders notları Yönetici Paneli üzerinden ilgili ders klasörüne yüklenir.
        2. Yönetici Paneli üzerinden RAG indeksi yenilenir.
        3. Kullanıcı RAG Arama sayfasında kaynak parçaları test eder.
        4. Kullanıcı LLM Destekli Cevap sayfasında soru sorar.
        5. Sistem ilgili kaynak parçaları bulur.
        6. LLM bu kaynaklara dayanarak açıklayıcı cevap üretir.
        """
    )