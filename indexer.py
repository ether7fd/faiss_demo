import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, DOCUMENT_PREFIX, DOCUMENTS_PATH, INDEX_PATH, MODEL_NAME

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


def main():
    print("=== インデックス作成 ===\n")

    print("[1/3] モデルを読み込み中...")
    model = SentenceTransformer(MODEL_NAME)

    print("[2/3] 文書をエンベディング中...")
    prefixed = [f"{DOCUMENT_PREFIX}{doc}" for doc in SAMPLE_DOCUMENTS]
    embeddings = model.encode(prefixed, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    print("[3/3] インデックスを保存中...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    DATA_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    DOCUMENTS_PATH.write_text(json.dumps(SAMPLE_DOCUMENTS, ensure_ascii=False, indent=2))

    print(f"\n完了！ {index.ntotal} 件の文書をインデックス化しました")
    print(f"  インデックス: {INDEX_PATH}")
    print(f"  文書リスト:   {DOCUMENTS_PATH}")


if __name__ == "__main__":
    main()
