import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_llm_answer(user_question, retrieved_item):
    """
    Kullanıcının sorusu ile veri setinden bulunan en yakın kaydı alır.
    LLM bu kaydı temel alarak daha doğal, öğretici bir cevap üretir.
    """

    context = f"""
Konu: {retrieved_item.get("topic", "")}
Seviye: {retrieved_item.get("level", "")}
Cevap tipi: {retrieved_item.get("type", "")}
Eşleşen soru: {retrieved_item.get("instruction", "")}
Temel cevap: {retrieved_item.get("output", "")}
"""

    instructions = """
Sen olasılık ve istatistik alanında Türkçe konuşan bir öğretici asistansın.

Kurallar:
1. Cevabı öğrencinin anlayacağı şekilde açıkla.
2. Gerektiğinde formül ver.
3. Gerektiğinde kısa örnek ver.
4. Verilen temel cevaba sadık kal.
5. Emin olmadığın bilgiyi uydurma.
6. Cevabı gereksiz uzatma.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=instructions,
        input=f"""
Kullanıcı sorusu:
{user_question}

Veri setinden bulunan bilgi:
{context}

Bu bilgiye dayanarak öğretici bir cevap üret.
"""
    )

    return response.output_text