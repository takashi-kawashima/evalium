import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from evalium.api.embeddings_api import EmbeddingClient
from evalium.dataset import Conversation, Conversations

# Load local.env if present
load_dotenv("local.env")


def find_data_folders(root: str) -> List[str]:
    folders = []
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "input.json")):
            folders.append(path)
    return folders


def add_embeddings(
    dataset: Conversation, client: EmbeddingClient, rating_threshold: float
):
    # for each example, if rating is above threshold, add embedding of agent_response to dataset.embeddings
    for i, example in dataset.df.iterrows():
        if dataset.embeddings.get(i) is not None:
            emb = dataset.embeddings[i]
            # print("already has embedding, using existing one")
            continue
        resp_text = example["agent_response"]
        if not isinstance(resp_text, str) or not resp_text or resp_text == "nan":
            continue
        if resp_text.startswith("ERROR:"):
            # print("Error response, skipping embedding")
            continue
        #        if (example['rating'] is None) or (np.isnan(example['rating'])) or (example['rating'] <= rating_threshold):
        #            continue
        # retrieve embeddings
        emb = client.embed_texts([resp_text])[0]
        #         enc = json.dumps(emb.tolist())
        dataset.embeddings[i] = emb


def build_index(data_dir: str, rating_threshold: float = 4.0):
    client = EmbeddingClient()

    conversations = Conversations.from_folder(
        data_dir
    )  # just to validate folder structure and load metadata

    folders = find_data_folders(data_dir)
    folder_basename = os.path.basename(data_dir)
    # smith = LangSmithIntegration()
    # create examples as intermediate data
    index_examples = []
    for key, conv in conversations.conversations.items():
        conv.apply_master_info(conversations.master)
        add_embeddings(dataset=conv, client=client, rating_threshold=rating_threshold)
        conv.save()
        conv_embeddings, ave_embeddings = conv.fetch_dataset_embeddings()
        # build index examples
        index_examples.append(
            {
                "inputs": conv.name,
                "embeddings": ave_embeddings,
                "metadata": conv.metadata,
            }
        )
        print(
            f"Processed conversation {conv.name}, user_message: {conv.metadata['user_message']}"
        )

    index_dataset = Conversation.from_examples(
        examples=index_examples, turn="", path=data_dir, dataset_name="index_dataset"
    )
    index_dataset.save()
    # if smith is not None:
    #     smith_dataset = smith.create_dataset_from_dummy(dataset=index_dataset)

    return index_dataset


def _find_golden_conversation(
    index_path: str, conversation_name: str
) -> Optional[Conversation]:
    for turn_path in glob.glob(os.path.join(index_path, "*")):
        if os.path.isdir(turn_path):
            conv_path = os.path.join(turn_path, conversation_name)
            if os.path.isdir(conv_path) and os.path.exists(
                os.path.join(conv_path, "input.json")
            ):
                return Conversation.from_folder(conv_path)
    return None


def _collect_embeddings(conv: Conversation):
    indices = []
    embs = []
    for i, _example in conv.df.iterrows():
        emb = conv.embeddings.get(i)
        if emb is not None:
            indices.append(i)
            embs.append(np.array(emb, dtype=np.float64))
    return indices, np.array(embs) if embs else np.empty((0, 0))


def _load_ok_follow_ups(
    index_path: str, conversation_name: str
) -> List[str]:
    """Load ok_follow_up_list from the master table for a given conversation."""
    master_file = os.path.join(index_path, "golden_data_master_table.xlsx")
    if not os.path.exists(master_file):
        return []
    master = pd.read_excel(master_file)
    rows = master.query(f'conversation == "{conversation_name}"')
    if rows.empty:
        return []

    all_follow_ups: List[str] = []
    for _, row in rows.iterrows():
        raw = row.get("ok_follow_up_list")
        if pd.isna(raw) or not str(raw).strip():
            continue
        parsed = list(csv.reader([str(raw)]))[0]
        all_follow_ups.extend(s.strip() for s in parsed if s.strip())
    return all_follow_ups


def _parse_follow_up_questions(raw: str) -> List[str]:
    """Parse the follow_up_questions column (JSON array) into a list of strings."""
    if not isinstance(raw, str) or not raw or raw == "nan":
        return []
    try:
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [str(q).strip() for q in questions if str(q).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _evaluate_follow_ups(
    dataset: Conversation,
    ok_follow_ups: List[str],
    client: EmbeddingClient,
) -> Dict[str, Any]:
    """Compute follow-up question similarity for every row in *dataset*.

    Returns a dict ready to be serialised as follow_up_results.json.
    """
    conversation_name = dataset.metadata.get("name", "")

    if not ok_follow_ups:
        return {
            "conversation_name": conversation_name,
            "ok_follow_ups": [],
            "per_row": {},
            "overall_average_best_similarity": None,
        }

    ok_embs = np.array(client.embed_texts(ok_follow_ups), dtype=np.float64)

    per_row: Dict[str, Any] = {}
    row_avg_scores: List[float] = []

    for i, example in dataset.df.iterrows():
        questions = _parse_follow_up_questions(example.get("follow_up_questions", ""))
        if not questions:
            continue

        gen_embs = np.array(client.embed_texts(questions), dtype=np.float64)
        sim = cosine_similarity(gen_embs, ok_embs)  # (n_gen, n_ok)

        best_matches = []
        for q_idx, q_text in enumerate(questions):
            best_ok_idx = int(np.argmax(sim[q_idx]))
            best_matches.append(
                {
                    "generated": q_text,
                    "best_ok": ok_follow_ups[best_ok_idx],
                    "similarity": round(float(sim[q_idx, best_ok_idx]), 6),
                }
            )

        avg_best = round(
            float(np.mean([m["similarity"] for m in best_matches])), 6
        )

        per_row[str(i)] = {
            "generated_questions": questions,
            "similarity_matrix": [
                [round(float(v), 6) for v in row] for row in sim.tolist()
            ],
            "best_matches": best_matches,
            "average_best_similarity": avg_best,
        }
        row_avg_scores.append(avg_best)

    overall = (
        round(float(np.mean(row_avg_scores)), 6) if row_avg_scores else None
    )

    return {
        "conversation_name": conversation_name,
        "ok_follow_ups": ok_follow_ups,
        "per_row": per_row,
        "overall_average_best_similarity": overall,
    }


def rank_query(
    index_path: str, dataset_folder: str, top_k: int = 5
) -> Dict[str, Any]:
    client = EmbeddingClient()

    # --- Load new dataset and ensure embeddings exist ---
    dataset = Conversation.from_folder(dataset_folder)
    add_embeddings(dataset=dataset, client=client, rating_threshold=-1.0)
    dataset.save()

    conversation_name = dataset.metadata.get("name")

    # --- Load the matching golden conversation (individual embeddings) ---
    golden_conv = _find_golden_conversation(index_path, conversation_name)
    if golden_conv is None:
        raise ValueError(
            f"Golden conversation '{conversation_name}' not found under {index_path}"
        )

    golden_indices, golden_embs = _collect_embeddings(golden_conv)
    new_indices, new_embs = _collect_embeddings(dataset)

    if golden_embs.size == 0 or new_embs.size == 0:
        raise ValueError("No embeddings found for golden or new dataset")

    # ========================================================
    # 20 × 20 similarity matrix
    # ========================================================
    sim_matrix = cosine_similarity(golden_embs, new_embs)
    sim_df = pd.DataFrame(
        sim_matrix,
        index=pd.Index(golden_indices, name="golden_run_index"),
        columns=pd.Index(new_indices, name="new_run_index"),
    )

    # ========================================================
    # Score 1: best (rating=5) vs new top_k
    # ========================================================
    best_indices = [
        i
        for i, ex in golden_conv.df.iterrows()
        if ex.get("rating") == 5.0 and i in golden_indices
    ]
    best_response_top_k: Dict[str, list] = {}
    best_avg_similarities: Dict[str, float] = {}
    for bi in best_indices:
        row_pos = golden_indices.index(bi)
        sims_row = sim_matrix[row_pos]
        best_avg_similarities[str(bi)] = round(float(np.mean(sims_row)), 6)
        top_k_pos = np.argsort(-sims_row)[:top_k]
        best_response_top_k[str(bi)] = [
            {"new_id": int(new_indices[j]), "score": round(float(sims_row[j]), 6)}
            for j in top_k_pos
        ]

    # ========================================================
    # Score 2: average similarity of full matrix
    # ========================================================
    average_similarity = round(float(np.mean(sim_matrix)), 6)

    # ========================================================
    # Score 3: golden average vector vs each new response
    # ========================================================
    index = Conversation.from_index(index_path)
    golden_avg_emb = np.array(index.embeddings[conversation_name]).reshape(1, -1)
    avg_vec_sims = cosine_similarity(golden_avg_emb, new_embs).flatten()
    avg_vector_ranking = sorted(
        [
            {"new_id": int(new_indices[j]), "score": round(float(avg_vec_sims[j]), 6)}
            for j in range(len(new_indices))
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    # ========================================================
    # Score 4: golden avg vector vs new avg vector
    # ========================================================
    new_avg_emb = np.mean(new_embs, axis=0, keepdims=True)
    avg_vs_avg_similarity = round(
        float(cosine_similarity(golden_avg_emb, new_avg_emb).flatten()[0]), 6
    )

    # ========================================================
    # Save results
    # ========================================================
    output_dir = os.path.join(dataset.path, "rank_results")
    os.makedirs(output_dir, exist_ok=True)

    sim_df.to_csv(
        os.path.join(output_dir, "similarity_matrix.csv"), encoding="utf-8_sig"
    )

    results = {
        "conversation_name": conversation_name,
        "best_response_top_k": best_response_top_k,
        "best_avg_similarity": best_avg_similarities,
        "average_similarity": average_similarity,
        "avg_vs_avg_similarity": avg_vs_avg_similarity,
        "average_vector_ranking": avg_vector_ranking,
    }
    with open(
        os.path.join(output_dir, "rank_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ========================================================
    # Follow-up question similarity
    # ========================================================
    ok_follow_ups = _load_ok_follow_ups(index_path, conversation_name)
    follow_up_results = _evaluate_follow_ups(dataset, ok_follow_ups, client)
    with open(
        os.path.join(output_dir, "follow_up_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(follow_up_results, f, indent=2, ensure_ascii=False)

    return results
