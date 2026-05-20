from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI

try:
    import streamlit as st
except Exception:
    st = None

try:
    from app.rag_answer import retrieve_context
except ModuleNotFoundError:
    from rag_answer import retrieve_context


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_secret(name: str, default: str = "") -> str:
    """
    Önce Streamlit Cloud secrets içinden okumaya çalışır.
    Yoksa yerel .env / ortam değişkenlerinden okur.
    """
    if st is not None:
        try:
            value = st.secrets.get(name, default)
            if value:
                return value
        except Exception:
            pass

    return os.getenv(name, default)


def build_context_text(contexts):
    parts = []

    for i, ctx in enumerate(contexts, start=1):
        parts.append(
            f"""
[KAYNAK {i}]
Dosya: {ctx["source"]}
Parça No: {ctx["chunk_id"]}
Benzerlik Skoru: {ctx["score"]:.4f}

Metin:
{ctx["text"]}
""".strip()
        )

    return "\n\n".join(parts)


def answer_with_llm(
    question: str,
    top_k: int = 3,
    model: str = "gpt-4.1-mini",
    course: str = "Genel",
):
    contexts = retrieve_context(
        question,
        top_k=top_k,
        course=course,
    )

    if not contexts:
        return {
            "question": question,
            "answer": (
                f"`{course}` dersi için indekslenmiş kaynak bulunamadı. "
                "Lütfen Yönetici Paneli’nden bu derse ait PDF/DOCX dosyası yükleyip "
                "RAG indeksini yenileyin."
            ),
            "contexts": [],
        }

    context_text = build_context_text(contexts)

    api_key = get_secret("OPENAI_API_KEY")

    if not api_key:
        return {
            "question": question,
            "answer": (
                "OPENAI_API_KEY bulunamadı. Yerelde `.env` dosyasına, "
                "Streamlit Cloud’da ise Secrets bölümüne OPENAI_API_KEY ekleyin."
            ),
            "contexts": contexts,
        }

    client = OpenAI(api_key=api_key)

    prompt = f"""
Sen bir üniversite düzeyinde olasılık, istatistik ve veri analizi ders asistanısın.

Aşağıdaki DERS NOTU PARÇALARINI kullanarak öğrencinin sorusuna Türkçe cevap ver.

Kurallar:
1. Sadece verilen ders notu parçalarına dayan.
2. Bilgi ders notlarında yoksa "Bu bilgi verilen ders notlarında açıkça yer almıyor." de.
3. Cevabı kısa, açık ve öğretici yaz.
4. Matematiksel formülleri Markdown uyumlu LaTeX biçiminde yaz.
5. Satır içi formülleri $...$ içinde yaz.
6. Ayrı satırdaki formülleri $$...$$ içinde yaz.
7. Cevap yapısı mümkünse şu sırada olsun:
   - Kısa Tanım
   - Formül
   - Açıklama
   - Örnek
   - Kaynaklar
8. Kaynaklar kısmında dosya adı ve parça numarasını mutlaka belirt.
9. Cevabı seçilen ders bağlamına uygun ver.
10. Ders notu parçasında formül bozuk karakterlerle geldiyse, anlam açıksa formülü düzgün matematiksel biçimde yaz.

SEÇİLEN DERS:
{course}

ÖĞRENCİ SORUSU:
{question}

DERS NOTU PARÇALARI:
{context_text}
""".strip()

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return {
        "question": question,
        "answer": response.output_text,
        "contexts": contexts,
    }


if __name__ == "__main__":
    print("LLM Destekli RAG Cevap Sistemi")
    print("Çıkmak için q yazın.")
    print("-" * 50)

    while True:
        question = input("\nSorunuz: ").strip()

        if question.lower() in ["q", "quit", "exit", "çıkış"]:
            print("Program kapatıldı.")
            break

        if not question:
            continue

        course = input("Ders seçiniz [Genel/Olasılık/İstatistik/SPSS/Regresyon]: ").strip()
        if not course:
            course = "Genel"

        result = answer_with_llm(
            question=question,
            course=course,
        )

        print("\nCEVAP")
        print("=" * 50)
        print(result["answer"])

        print("\nKULLANILAN KAYNAK PARÇALAR")
        print("=" * 50)

        for i, ctx in enumerate(result["contexts"], start=1):
            print(
                f"{i}. Kaynak: {ctx['source']} | "
                f"Parça: {ctx['chunk_id']} | "
                f"Skor: {ctx['score']:.4f}"
            )