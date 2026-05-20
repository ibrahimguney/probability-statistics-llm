# Olasılık ve İstatistik LLM

Bu proje, olasılık ve istatistik dersleri için geliştirilmiş Streamlit tabanlı bir akademik yapay zekâ asistanıdır. Sistem, PDF ve Word ders notlarını okuyarak RAG yöntemiyle ilgili kaynak parçaları bulur ve OpenAI destekli LLM ile kaynaklara dayalı açıklayıcı cevaplar üretir.

## Temel Özellikler

- PDF ve DOCX ders notu yükleme
- Ders bazlı klasör yapısı
- RAG indeksleme
- Ders bazlı kaynak parça arama
- LLM destekli akademik cevap üretme
- Yönetici paneli
- Yönetici şifresi ile koruma
- Sohbet geçmişi
- Streamlit Cloud uyumu
- OpenAI API anahtarını güvenli biçimde Secrets üzerinden kullanma

## Kullanılan Teknolojiler

- Python
- Streamlit
- OpenAI API
- scikit-learn
- pypdf
- python-docx
- pandas
- python-dotenv

## Proje Yapısı

```text
probability-statistics-llm/
│
├── app/
│   ├── streamlit_app.py
│   ├── rag_answer.py
│   └── llm_rag_answer.py
│
├── docs/
│   └── raw/
│       ├── olasilik/
│       ├── istatistik/
│       ├── spss/
│       ├── regresyon/
│       └── arastirma_yontemleri/
│
├── data/
│   └── rag_index/
│
├── ingest.py
├── search_rag.py
├── requirements.txt
├── README.md
└── .gitignore