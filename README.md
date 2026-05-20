# Probability and Statistics RAG Assistant

Bu proje, olasılık ve istatistik konularında Türkçe destekli bir öğretici yapay zekâ asistanı geliştirmek amacıyla hazırlanmıştır.

Uygulama; yapılandırılmış soru-cevap veri seti, OpenAI embedding modeli, RAG yaklaşımı ve Streamlit arayüzü kullanarak çalışır.

## Özellikler

- Olasılık ve istatistik konularında öğretici cevap üretimi
- Embedding tabanlı kaynak arama
- RAG destekli cevap üretimi
- OpenAI LLM bağlantısı
- Sohbet geçmişi
- Kaynak gösterimi
- Cevap uzunluğu seçimi
- Öğrenci seviyesi seçimi
- Soru üretici modül
- Çoktan seçmeli, klasik, çözümlü ve kısa cevaplı soru üretimi
- Streamlit web arayüzü

## Proje Yapısı

```text
probability-statistics-llm/
│
├── app/
│   ├── build_embeddings.py
│   ├── llm_helper.py
│   ├── retriever.py
│   └── streamlit_app.py
│
├── data/
│   ├── probability_statistics_qa.jsonl
│   ├── probability_statistics_qa_structured.jsonl
│   ├── qa_embeddings.npy
│   └── qa_metadata.jsonl
│
├── requirements.txt
├── README.md
└── .gitignore