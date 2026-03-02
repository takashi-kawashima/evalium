# Score Dashboard

`scripts/rank_all.sh` 実行後のランキング結果を可視化するノートブック。

### Agent Response スコア定義
| カラム | 説明 |
|--------|------|
| `best_id` | Golden best response (rating=5) の run index |
| `best_top1_new_id` | その golden best に最も類似していた new run の ID |
| `best_top1_score` | そのコサイン類似度スコア |
| `average_similarity` | (Score 2) Golden × New 全ペアの類似度行列の平均 |
| `avg_vs_avg_similarity` | (Score 4) Golden 平均ベクトル vs New 平均ベクトルのコサイン類似度 |
| `best_avg_similarity` | (Score 1) Best (rating=5) golden response → 全 new responses の平均類似度 |


### Follow-up Question スコア定義

各 run には 3 つの follow-up question が生成される。マスターの `ok_follow_up_list` に含まれる正解 follow-up とコサイン類似度で比較する。

| カラム | 説明 |
|--------|------|
| `n_ok_follow_ups` | マスターに登録されている OK follow-up question の数 |
| `per_run_max_sims` | 各 run ごとに、生成された 3 つの follow-up question それぞれの最大類似度をリスト表示。例: `{"run0": [0.5, 0.4, 0.3], "run1": [0.2, 0.1, 0.9]}` |
| `follow_up_score` | 各 run で 3 つの最大類似度のうち **最大値** → 全 run の平均。例: run0 max=0.5, run1 max=0.9 → (0.5+0.9)/2=0.7 |
| `follow_up_avg_similarity` | 各 run で 3 つの最大類似度の **平均** → 全 run の平均。例: run0 mean=0.4, run1 mean=0.4 → (0.4+0.4)/2=0.4 |
| `best_run_id` | agent_response で golden best (rating=5) に最も類似した new run の ID |
| `best_run_fu_sims` | その best run の 3 つの follow-up の最大類似度リスト。例: `[0.5, 0.4, 0.3]` |
| `best_run_fu_avg` | `best_run_fu_sims` の平均。例: mean(0.5, 0.4, 0.3) = 0.4 |

### Token & Latency Stats 定義

各 conversation の `examples.xlsx` から run ごとの統計量を算出し、平均を表示する。

| カラム | 説明 |
|--------|------|
| `avg_time_seconds` | 全 run の平均レイテンシ（秒） |
| `avg_total_tokens` | 全 run の平均トータルトークン数 |
| `avg_prompt_tokens` | 全 run の平均プロンプトトークン数 |
| `avg_completion_tokens` | 全 run の平均コンプリーショントークン数 |

### Baseline との比較 (Diff テーブル)

`compare_scores` / `compare_follow_up_scores` / `compare_stats` は、各スコアの **差分 (target − baseline)** を表示し、閾値に基づいて3段階で色分けする。

#### Similarity 系 (higher is better)
| 色 | 条件 | 意味 |
|----|------|------|
| **Green** | diff >= upper | 大きく改善 |
| **Yellow** | lower <= diff < upper | 微変化 / ほぼ同等 |
| **Red** | diff < lower | 大きく悪化 |

#### Stats 系 (lower is better)
| 色 | 条件 | 意味 |
|----|------|------|
| **Green** | diff <= −upper | 大きく改善（減少） |
| **Yellow** | −upper < diff < |lower| | 微変化 / ほぼ同等 |
| **Red** | diff >= |lower| | 大きく悪化（増加） |

**Gray** = baseline に該当なし (N/A)。`upper` / `lower` は各関数の引数で変更可能（デフォルト: similarity upper=0.05, lower=-0.05 / stats upper=5.0, lower=-5.0）。