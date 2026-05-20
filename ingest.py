from pathlib import Path
import json
import pickle

from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer


RAW_DIR = Path("docs/raw")
DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "rag_index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)

    return "\n".join(texts)


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    paragraphs = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def split_text(text: str, chunk_size: int = 700, overlap: int = 120):
    text = " ".join(text.split())
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def load_documents():
    documents = []

    for path in RAW_DIR.rglob("*"):
        if path.suffix.lower() == ".pdf":
            text = read_pdf(path)
        elif path.suffix.lower() == ".docx":
            text = read_docx(path)
        else:
            continue

        chunks = split_text(text)

        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "source": str(path.relative_to(RAW_DIR)),
                    "chunk_id": i,
                    "text": chunk,
                }
            )

    return documents


def main():
    print("Dokümanlar okunuyor...")
    documents = load_documents()

    if not documents:
        print("Hiç doküman bulunamadı. Lütfen docs/raw klasörünü kontrol edin.")
        return

    print(f"Toplam parça sayısı: {len(documents)}")

    texts = [doc["text"] for doc in documents]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(texts)

    with open(INDEX_DIR / "documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    with open(INDEX_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(INDEX_DIR / "matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)

    print("RAG indeksi başarıyla oluşturuldu.")
    print(f"Kayıt yeri: {INDEX_DIR}")

def rebuild_index():
    main()

if __name__ == "__main__":
    main()