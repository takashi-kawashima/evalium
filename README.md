LLM評価ツール（簡易）

概要
- 与えられた人手評価済みデータセットから「良い回答」の埋め込みを作り、新規ユーザー入力に対してコサイン類似度でランキングを行います。

構成
- 言語: Python
- 主要ライブラリ: langchain, langgraph（任意）, openai, pandas, numpy

使い方（簡易）
1. 環境変数を設定（OpenAIまたはAzure OpenAI）:

   - OpenAI:
     - `OPENAI_API_KEY`

   - Azure OpenAI:
     - `AZURE_OPENAI_API_KEY`
     - `AZURE_OPENAI_API_BASE`（例: https://your-resource.openai.azure.com/）
     - `AZURE_OPENAI_DEPLOYMENT_NAME`

    - Rakuten Gateway (社内GW):
      - `RAKUTEN_AI_GATEWAY_KEY`（必須）
      - `RAKUTEN_AI_GATEWAY_BASE`（オプション、デフォルト: https://api.ai.public.rakuten-it.com/openai/v1 ）

2. データ配置: 各入力はフォルダ単位で配置します。フォルダ内に必ず `input.json`（ユーザー入力）を置き、人手評価結果のExcelファイル（*.xlsx）を同じフォルダに置いてください。

3. 依存関係をインストール:

```bash
pip install -r requirements.txt
```

4. インデックス作成例:

```bash
python -m evalium.cli build-index --data-dir data --out embeddings.npz --threshold 4
```

5. ランキング例:

```bash
python -m evalium.cli rank --embeddings embeddings.npz --query "ユーザー入力テキスト" --top-k 5
```

注意
- LangChainの概念に沿った設計を心がけていますが、埋め込み呼び出しはOpenAI Python SDKを直接使い簡潔にしています。
- `langgraph` を使った簡易オーケストレーション例はオプションで用意しました（要インストール）。
# evalium
Evaluation platform for LLM and Agent workflow
