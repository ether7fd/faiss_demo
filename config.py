from pathlib import Path

MODEL_NAME = "cl-nagoya/ruri-v3-70m"
QUERY_PREFIX = "検索クエリ: "
DOCUMENT_PREFIX = "検索文書: "

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "index.faiss"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
