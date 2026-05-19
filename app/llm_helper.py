import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_retrieved_context(retrieved_items):
    """
    retrieved_items bir listedir.
    Her eleman veri setinden gelen bir sözlük/dict kaydıdır.
    """

    context_parts = []

    for i, item in enumerate(retrieved_items, start=1):
        context_parts.append(
            f"""
Kaynak {i}
Konu: {item.get("topic", "")}
Seviye: {item.get("level", "")}
Cevap tipi: {item.get("type", "")}
Eşleşen soru: {item.get("instruction", "")}
Temel cevap: {item.get("output", "")}
Benzerlik skoru: {item.get("score", "")}
"""
        )

    return "\n".join(context_parts)


def generate_llm_answer(user_question, retrieved_items):
    """
    Kullanıcı sorusu ile embedding aramasından gelen kaynak listesini alır.
    LLM bu kaynaklara dayanarak öğretici cevap üretir.
    """

    if not isinstance(retrieved_items, list):
        retrieved_items = [retrieved_items]

    context = format_retrieved_context(retrieved_items)

    instructions = """
Sen olasılık ve istatistik alanında Türkçe konuşan bir öğretici asistansın.

Kurallar:
1. Cevabı öğrencinin anlayacağı şekilde açıkla.
2. Gerektiğinde formül ver.
3. Gerektiğinde kısa örnek ver.
4. Sadece verilen kaynak bilgilerden yararlan.
5. Kaynaklarda olmayan bilgiyi uydurma.
6. Eğer kaynak yetersizse bunu açıkça söyle.
7. Cevabı gereksiz uzatma.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=instructions,
        input=f"""
Kullanıcı sorusu:
{user_question}

Veri setinden bulunan kaynaklar:
{context}

Bu kaynaklara dayanarak öğretici bir cevap üret.
"""
    )

    return response.output_text


def generate_questions(topic, level, question_type, question_count):
    """
    Seçilen konu, seviye ve soru türüne göre öğretici sorular üretir.
    """

    instructions = """
Sen olasılık ve istatistik alanında üniversite düzeyinde soru hazırlayan bir öğretim asistanısın.

Kurallar:
1. Sorular açık, anlaşılır ve akademik düzeyde olsun.
2. Her soru için doğru cevap ver.
3. Çözümlü soru istenirse adım adım çözüm yaz.
4. Çoktan seçmeli soru istenirse A, B, C, D seçenekleri üret.
5. Doğru seçeneği açıkça belirt.
6. Gerektiğinde formül kullan.
7. Türkçe yaz.
"""

    prompt = f"""
Aşağıdaki özelliklere göre olasılık ve istatistik soruları hazırla:

Konu: {topic}
Seviye: {level}
Soru türü: {question_type}
Soru sayısı: {question_count}

Çıktı formatı:

1. Soru:
...

Doğru cevap:
...

Çözüm / Açıklama:
...

Eğer soru türü çoktan seçmeli ise seçenekleri şu biçimde ver:

A)
B)
C)
D)

Doğru seçenek:
...
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=instructions,
        input=prompt
    )

    return response.output_text