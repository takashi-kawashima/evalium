import os
import json
import glob
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import openai
from datetime import datetime, timezone
from langsmith_integration import LangSmithIntegration
# Load local.env if present
load_dotenv("local.env")


class EmbeddingClient:
    """Embedding client that uses Rakuten Gateway via LangChain-style embeddings.
import openai

    This tool is configured to operate in GW-only mode: `RAKUTEN_AI_GATEWAY_KEY` must be set.
    It will attempt to use `langchain_openai.OpenAIEmbeddings` (or `AzureOpenAIEmbeddings`),
    falling back to `langchain.embeddings.OpenAIEmbeddings` where available.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        rak_key = os.getenv("RAKUTEN_AI_GATEWAY_KEY")
        if not rak_key:
            raise RuntimeError("Rakuten Gateway only: set RAKUTEN_AI_GATEWAY_KEY in local.env")

        self._rakuten_key = rak_key
        # Default endpoints based on examples provided
        self._rakuten_base_openai = os.getenv("RAKUTEN_AI_GATEWAY_OPENAI_BASE")
        self._rakuten_base_azure = os.getenv("RAKUTEN_AI_GATEWAY_AZURE_BASE")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0,))

        # Try 0: openai.OpenAI (user example)
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self._rakuten_key,
                base_url=self._rakuten_base_openai
            )
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            return np.array([d.embedding for d in response.data])
        except Exception:
            pass

        # Try 1: langchain_openai.OpenAIEmbeddings (user example)
        try:
            from langchain_openai import OpenAIEmbeddings

            emb = OpenAIEmbeddings(
                model=self.model,
                base_url=self._rakuten_base_openai,
                api_key=self._rakuten_key,
            )
            vecs = emb.embed_documents(texts)
            return np.array(vecs, dtype=np.float32)
        except Exception:
            pass

        # Try 2: langchain_openai.AzureOpenAIEmbeddings (user example)
        try:
            from langchain_openai import AzureOpenAIEmbeddings

            emb = AzureOpenAIEmbeddings(
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                azure_endpoint=self._rakuten_base_azure,
                api_key=self._rakuten_key,
            )
            vecs = emb.embed_documents(texts)
            return np.array(vecs, dtype=np.float32)
        except Exception:
            pass

        # Try 3: langchain built-in OpenAIEmbeddings as fallback
        try:
            from langchain.embeddings import OpenAIEmbeddings as LCOpenAIEmbeddings

            emb = LCOpenAIEmbeddings(
                openai_api_key=self._rakuten_key,
                openai_api_base=self._rakuten_base_openai,
                model=self.model,
            )
            vecs = emb.embed_documents(texts)
            return np.array(vecs, dtype=np.float32)
        except Exception as e:
            # Final fallback: direct OpenAI-compatible API call to Rakuten GW
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self._rakuten_key, base_url=self._rakuten_base_openai)
                resp = client.embeddings.create(model=self.model, input=texts)
                return np.array([d.embedding for d in resp.data], dtype=np.float32)
            except Exception as e2:
                raise RuntimeError(
                    "Failed to create embeddings via Rakuten Gateway (langchain and direct OpenAI failed)."
                ) from e2


def find_data_folders(root: str) -> List[str]:
    folders = []
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "input.json")):
            folders.append(path)
    return folders


def load_input_json(folder: str) -> dict:
    p = os.path.join(folder, "input.json")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def find_excel_file(folder: str) -> Optional[str]:
    files = glob.glob(os.path.join(folder, "*.xlsx"))
    return files[0] if files else None

def load_ratings_from_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def build_reference_embeddings(data_dir: str, out_path: str, rating_threshold: float = 4.0, assume_all: bool = False):
    client = EmbeddingClient()
    folders = find_data_folders(data_dir)
    smith = LangSmithIntegration()

    for folder in folders:
        input_json = load_input_json(folder)
        excel = find_excel_file(folder)
        if not excel:
            continue
        df = load_ratings_from_excel(excel)
        # detect likely columns
        text_col = "agent_response"
        rating_col = "rating"
        if rating_col not in df.columns:
            rating_col = None 
        df['config_name'] = input_json['config_name']
        df['shop_id'] = input_json['shop_id']
        df['prompt_date'] = input_json['prompt_date']
        df['user_message'] = input_json['user_message']
        dataset_name=f"{input_json['config_name']}_shop{input_json['shop_id']}_{input_json['prompt_date']}"
        try:
            smith.create_dataset(
                dataset_name=dataset_name,
                description=f"Dataset for folder {folder} with input and responses",
                df=df,
                input_keys=["user_message","run_index","config_name","shop_id","prompt_date"],
                output_keys=[text_col,"follow_up_questions","tools_and_arguments","iteration_count","time_seconds","total_tokens","prompt_tokens","completion_tokens"]
            )
        except Exception as e:
            print("Error saving/sending metadata:", e)
    
        examples = smith.list_examples(dataset_name=dataset_name)
        for example in examples:
            print(f"LangSmith dataset: {example.inputs['user_message']} (id: {example.id})")
            if example.outputs.get("embedding"):
                print("already has embedding, skipping")
                continue
            resp_text = example.outputs[text_col]
            emb = client.embed_texts([resp_text])[0]
            outputs = example.outputs
            outputs["embedding"] = emb
            smith.update_example(example_id=example.id, outputs=outputs)

        # for _, row in df.iterrows():
        #     resp_text = str(row[text_col])
        #     if rating_col is None or assume_all:
        #         # include all responses when no rating column or assume_all flag set
        #         emb = client.embed_texts([resp_text])[0]
        #         all_embeddings.append(emb)
        #         all_meta.append({"folder": os.path.basename(folder), "response": resp_text, "rating": None, "input": input_json})
        #     else:
        #         try:
        #             rating = float(row[rating_col])
        #         except Exception:
        #             continue
        #         if rating >= rating_threshold:
        #             emb = client.embed_texts([resp_text])[0]
        #             all_embeddings.append(emb)
        #             all_meta.append({"folder": os.path.basename(folder), "response": resp_text, "rating": rating, "input": input_json})
        break

    # if all_embeddings:
    #     X = np.stack(all_embeddings)
    # else:
    #     X = np.zeros((0, 1), dtype=np.float32)

    # # persist embeddings + meta
    # np.savez(out_path, embeddings=X, meta=all_meta)

    # # write metadata summary and try send to LangSmith
    # try:
    #     meta = {
    #         "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    #         "data_dir": data_dir,
    #         "n_refs": len(all_meta),
    #         "out": out_path,
    #         "rating_threshold": rating_threshold,
    #         "assume_all": bool(assume_all),
    #         "model": client.model,
    #     }
    #     meta_path = smith.save_metadata(meta, os.path.join(os.path.dirname(out_path) or ".", "artifacts"))
    #     _sent = smith.try_send_to_langsmith(meta, X, out_path)
    #     meta["langsmith_sent"] = bool(_sent)
    # except Exception as e:
    #     print("Error saving/sending metadata:", e)
    #     meta = {"n_refs": len(all_meta), "out": out_path}

    return 


def load_index(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    
    return {"embeddings": data["embeddings"], "meta": data["meta"]}


def rank_query(embeddings_path: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    idx = load_index(embeddings_path)
    refs = idx["embeddings"]
    meta = list(idx["meta"])
    if refs.size == 0:
        return []

    client = EmbeddingClient()
    q_emb = client.embed_texts([query])[0]

    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(refs, q_emb.reshape(1, -1)).reshape(-1)
    order = np.argsort(-sims)[:top_k]
    results = []
    for i in order:
        results.append({"score": float(sims[i]), "meta": meta[i]})
    return results
