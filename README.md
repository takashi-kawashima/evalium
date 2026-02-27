# evalium: LLM応答ランキングツール

## 概要

人手評価済みまたは生のLLM応答データセットから埋め込みを生成→新規クエリとのコサイン類似度で上位応答をランキングするシンプルなツール。

**主要な特徴:**
- Rakuten AI Gateway 対応（OpenAI互換エンドポイント）
- 埋め込みメタデータから LangSmith への自動トレース送信（オプション）
- 環境変数で簡単設定（`local.env` サポート）

## 前提・環境要件

- Python 3.9+
- OpenAI SDK（`openai>=1.0`）
- `langchain-openai` または `langchain` （埋め込みのフォールバック用）
- `pandas`, `numpy`, `scikit-learn`

## セットアップ

### 1. 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数を設定（`local.env` に記述）

**Rakuten Gateway を使う場合（推奨・社内GW）:**

```env
RAKUTEN_AI_GATEWAY_KEY="your-rakuten-api-key"
RAKUTEN_AI_GATEWAY_OPENAI_BASE="https://api.ai.public.rakuten-it.com/openai/v1"
LANGSMITH_API_KEY="your-langsmith-api-key"
```

**OpenAI を使う場合:**

```env
OPENAI_API_KEY="sk-proj-..."
```

**Azure OpenAI を使う場合:**

```env
AZURE_OPENAI_API_KEY="your-azure-key"
AZURE_OPENAI_API_BASE="https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2023-05-15"
```

### 3. データ構造

プロジェクトの `data/` フォルダに以下のように配置します：

```
data/
├──master.xlsx                  # マスターファイル
└── 1/                          # フォルダごとに1つの入力 turnに対応したりする
    ├── input.json              # ユーザー入力（JSON）
    └── responses.xlsx          # 応答一覧（Excel: 列は agent_response, rating など）
```

**master.xlsx  の列:**
- `conversation` （必須）: キーとなるexamplesフォルダの名称 例："こんにちは_20回_free_v02_100064_20260218_162721"
- `case` （必須）: ケースID 同じtopicに対して複数選択された場合のインデックス
- `best_response_id` : 最良のexampleID
- `ok_response_id` : 次点のexampleID
- `turn` : 何回目の問か

**input.json の例:**
```json
{
  "user_message": "こんにちは",
  "shop_id": 100064,
  "config_name": "v02"
}
```

**responses.xlsx  の列:**
- `run_index` （必須）: インデックス
- `agent_response` （必須）: LLMの応答テキスト
- `rating` （オプション）: 人手評価スコア（閾値のデフォルト：4.0以上を良応答として扱う）:masterファイルから自動で付与される

## 使用方法

### インデックス作成

```bash
python -m evalium.cli build-index --data-dir "my_dataset" --threshold 4.0
```

**オプション:**
- `--data-dir`: 入力フォルダ（複数のCOnversation 階層を含む）
- `--threshold`: 評価スコアの下限値（デフォルト：4.0）

**実行例:**
```bash
# 全データをインデックス化して LangSmith に保存
python -m evalium.cli build-index --data-dir "data/golden-dataset" --threshold -1.0
```

**出力:**
- LangSmith にデータセットを作成・保存
- ターミナルに `Index built, dataset id: <dataset-uuid>` を表示

### ランキング（クエリから最も類似した応答を検索）

```bash
python -m evalium.cli rank --index "index-dataset-folder" --dataset "target-dataset-folder" --top-k 5
```
**オプション:**
- `--index`: indexされたデータセットのフォルダ（embeddings.csvが存在するもの）
- `--dataset`: 類似度計算、ランキング対象となるデータセットのフォルダ

**実行例:**
```bash
python -m evalium.cli rank --index "data/goldendatasetv1" --dataset "data/new_dataset" --top-k 10
```



**出力形式:**
```
Ranking results:
Rank 1: id=abc-123 , score=0.8234 , text=こんにちは！データ分析アシスタント...
Rank 2: id=def-456 , score=0.7891 , text=ご質問ありがとうございます...
```

## アーキテクチャ

- **`evalium/evaluator.py`**: コア機能
  - `EmbeddingClient`: Rakuten Gateway/OpenAI/Azure への埋め込み呼び出し
  - `build_reference_embeddings()`: データセットを読み込んで LangSmith にデータセット化
  - `fill_embeddings_if_missing()`: 埋め込みが無い場合は生成
  - `rank_query()`: LangSmith データセット内でコサイン類似度によるランキング

- **`evalium/langsmith_integration.py`**: 実験管理
  - メタデータ保存と LangSmith への送信（ベストエフォート）

- **`evalium/cli.py`**: CLIインターフェース
  - `build-index`: インデックス作成
  - `rank`: ランキング実行

## LangSmith 統合

`LANGSMITH_API_KEY` が設定されている場合、ビルド実行時に `artifacts/metadata-<hash>.json` を自動生成し、LangSmith へ送信を試みます（失敗時も動作継続）。

## トラブルシューティング

**SSL エラー:**
- 社内ファイアウォール/プロキシ設定を確認してください

**ランキング結果が低スコア:**
- クエリと候補テキストの意味が大きく異なっています
- `--top-k` を増やして確認してください

## サンプル実行

```bash
pip install -r requirements.txt
python -m evalium.cli build-index --data-dir data --out "golden_v1" --threshold 4.0
python -m evalium.cli rank --index "test-index" --dataset "golden_v1" --top-k 3
```
# evalium
Evaluation platform for LLM and Agent workflow
