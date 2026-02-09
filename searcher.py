import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import DOCUMENTS_PATH, INDEX_PATH, MODEL_NAME, QUERY_PREFIX, SCORE_THRESHOLD


def search(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    documents: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    query_embedding = model.encode(
        [f"{QUERY_PREFIX}{query}"], normalize_embeddings=True
    )
    query_embedding = np.array(query_embedding, dtype=np.float32)

    scores, indices = index.search(query_embedding, top_k)

    return [
        (documents[idx], float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx != -1 and score >= SCORE_THRESHOLD
    ]


def main():
    print("=== FAISS テキスト検索 ===\n")

    print("[1/2] モデルを読み込み中...")
    model = SentenceTransformer(MODEL_NAME)

    print("[2/2] インデックスを読み込み中...")
    index = faiss.read_index(str(INDEX_PATH))
    documents = json.loads(DOCUMENTS_PATH.read_text())

    print(f"  {index.ntotal} 件の文書を読み込みました")
    print("\n検索クエリを入力してください（終了: q）\n")

    while True:
        try:
            query = input("検索> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not query:
            continue
        if query.lower() == "q":
            print("終了します。")
            break

        results = search(query, model, index, documents)

        if not results:
            print(f"\n「{query}」に該当する文書が見つかりませんでした\n")
            continue

        print(f"\n「{query}」の検索結果（{len(results)} 件）:")
        print("-" * 60)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"  {rank}. [スコア: {score:.4f}] {doc}")
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
