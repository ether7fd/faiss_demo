import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# サンプル文書データ
SAMPLE_DOCUMENTS = [
    "Pythonは汎用プログラミング言語で、機械学習やデータ分析に広く使われています",
    "JavaScriptはウェブブラウザ上で動作するプログラミング言語です",
    "機械学習は人工知能の一分野で、データからパターンを学習します",
    "深層学習はニューラルネットワークを多層に重ねた機械学習手法です",
    "自然言語処理はコンピュータがテキストを理解・生成する技術です",
    "ベクトル検索は高次元ベクトル空間で類似性を計算する手法です",
    "FAISSはFacebookが開発した高速な類似ベクトル検索ライブラリです",
    "データベースは構造化されたデータを効率的に保存・検索するシステムです",
    "クラウドコンピューティングはインターネット経由でコンピュータリソースを提供します",
    "コンテナ技術はアプリケーションを隔離された環境で実行する仕組みです",
    "REST APIはHTTPプロトコルを使ったWebサービスの設計原則です",
    "GitはソースコードのバージョンManagementに使われる分散型管理システムです",
    "Dockerはコンテナ型仮想化プラットフォームで、環境構築を容易にします",
    "Kubernetesはコンテナオーケストレーションツールで、大規模なデプロイを管理します",
    "強化学習はエージェントが試行錯誤を通じて最適な行動を学ぶ手法です",
    "画像認識はコンピュータビジョンの一分野で、画像内の物体を識別します",
    "推薦システムはユーザーの好みに基づいてアイテムを提案するシステムです",
    "全文検索エンジンはテキストデータを高速に検索するための技術です",
    "埋め込み表現はテキストや画像を数値ベクトルに変換する技術です",
    "トランスフォーマーは自然言語処理で広く使われるニューラルネットワークアーキテクチャです",
]


def build_index(
    model: SentenceTransformer, documents: list[str]
) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    """文書をエンベディングしてFAISSインデックスを構築する"""
    print(f"  {len(documents)} 件の文書をエンベディング中...")
    prefixed = [f"検索文書: {doc}" for doc in documents]
    embeddings = model.encode(prefixed, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"  インデックス構築完了 (次元数: {dimension}, 文書数: {index.ntotal})")
    return index, embeddings


def search(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    documents: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """クエリに類似した文書を検索する"""
    query_embedding = model.encode([f"検索クエリ: {query}"], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((documents[idx], float(score)))
    return results


def main():
    print("=== FAISS テキスト検索デモ ===\n")

    print("[1/2] モデルを読み込み中...")
    model = SentenceTransformer("cl-nagoya/ruri-v3-70m")

    print("[2/2] インデックスを構築中...")
    index, _ = build_index(model, SAMPLE_DOCUMENTS)

    print("\n準備完了！検索クエリを入力してください（終了: q）\n")

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

        results = search(query, model, index, SAMPLE_DOCUMENTS)

        print(f"\n「{query}」の検索結果（上位 {len(results)} 件）:")
        print("-" * 60)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"  {rank}. [スコア: {score:.4f}] {doc}")
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
