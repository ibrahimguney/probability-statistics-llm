from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI

try:
    from app.rag_answer import retrieve_context
except ModuleNotFoundError:
    from rag_answer import retrieve_context


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


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


def answer_with_llm(question: str, top_k: int = 3, model: str = "gpt-4.1-mini", course: str = "Genel"):
    contexts = retrieve_context(question, top_k=top_k, course=course)
    context_text = build_context_text(contexts)

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "question": question,
            "answer": (
                "OPENAI_API_KEY bulunamadı. Lütfen proje ana klasöründe .env dosyasına "
                "OPENAI_API_KEY değerini ekleyin."
            ),
            "contexts": contexts,
        }

    client = OpenAI()

    prompt = f"""
Sen bir üniversite düzeyinde olasılık ve istatistik ders asistanısın.

Aşağıdaki DERS NOTU PARÇALARINI kullanarak öğrencinin sorusuna Türkçe cevap ver.

Kurallar:
1. Sadece verilen ders notu parçalarına dayan.
2. Bilgi ders notlarında yoksa "Bu bilgi verilen ders notlarında açıkça yer almıyor." de.
3. Cevabı kısa, açık ve öğretici yaz.
4. Matematiksel formülleri Markdown uyumlu LaTeX biçiminde yaz.
5. Satır içi formülleri $...$ içinde yaz.
6. Ayrı satırdaki formülleri $$...$$ içinde yaz.
7. Cevap yapısı şu sırada olsun:
   - Kısa Tanım
   - Formül
   - Açıklama
   - Örnek
   - Kaynaklar
8. Kaynaklar kısmında dosya adı ve parça numarasını mutlaka belirt.
9. Cevabı seçilen ders bağlamına uygun ver.

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

        result = answer_with_llm(question)

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