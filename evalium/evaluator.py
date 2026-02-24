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
from sklearn.metrics.pairwise import cosine_similarity
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

def fill_embeddings_if_missing(smith: LangSmithIntegration, dataset_name: str, client: EmbeddingClient):
    examples = smith.list_examples(dataset_name=dataset_name)
    for example in examples:
        resp_text = example.outputs.get("agent_response")
        if resp_text.startswith("ERROR:"):
            print("Error response, skipping embedding")
            continue
        if example.outputs.get("embedding"):
            print("already has embedding, using existing one")
            continue
        if not resp_text:
            continue
        emb = client.embed_texts([resp_text])[0]
        outputs = example.outputs
        outputs["embedding"] = emb
        smith.update_example(example_id=example.id, outputs=outputs)

def build_reference_embeddings(data_dir: str, out_path: str, rating_threshold: float = 4.0, assume_all: bool = False):
    client = EmbeddingClient()
    folders = find_data_folders(data_dir)
    folder_basename = os.path.basename(data_dir)
    smith = LangSmithIntegration()

    for folder in folders:
        dataset_basename = os.path.basename(folder)
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
        dataset_name=f"{dataset_basename}"
        try:
            smith.create_dataset(
                dataset_name=dataset_name,
                description=f"Dataset for folder {folder} with input and responses",
                df=df,
                input_keys=["user_message","run_index","config_name","shop_id","prompt_date"],
                output_keys=[text_col,"follow_up_questions","tools_and_arguments","iteration_count","time_seconds","total_tokens","prompt_tokens","completion_tokens",rating_col]
            )
        except Exception as e:
            print("Error saving/sending metadata:", e)
        new_examples = []
        all_emb = []
        fill_embeddings_if_missing(smith=smith, dataset_name=dataset_name, client=client)
        examples = smith.list_examples(dataset_name=dataset_name)
        for example in examples:
            print(f"LangSmith dataset: {example.inputs['user_message']} (id: {example.id})")
            if example.outputs.get("embedding"):
                print("already has embedding, using existing one")
                emb = example.outputs["embedding"]
            # check if rating is above threshold (if rating exists and assume_all is False)
            rating = example.outputs.get(rating_col)
            if rating is not None and rating > rating_threshold:
                all_emb.append(emb)
        
        if all_emb:
            all_emb = np.array(all_emb, dtype=np.float32)
            # modify all_emb is average vector of collection of all_emb, and normalize to unit vector below:
            all_emb = np.mean(all_emb, axis=0, keepdims=False)

            new_examples.append({
                "inputs": {"user_message": input_json['user_message'] },
                "outputs": {"embedding": all_emb},
                "metadata": {"user_message": input_json['user_message'],"config_name": input_json['config_name'],"shop_id": input_json['shop_id'],"prompt_date": input_json['prompt_date'] },
            })
        break

    new_dataset = smith.client.create_dataset(
        dataset_name=f"Indexed_dataset_{folder_basename}",
        description="Indexed dataset trained by goldern dataset",
    )
    smith.client.create_examples(
        dataset_id=new_dataset.id,
        examples=new_examples
        )
    return new_dataset

# def rank_query(dataset_name:str, query:str, top_k:int=5):


def load_index(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    
    return {"embeddings": data["embeddings"], "meta": data["meta"]}


def rank_query(index: str, dataset: str, top_k: int = 5) -> List[Dict[str, Any]]:
    client = EmbeddingClient()
    smith = LangSmithIntegration()
    
    fill_embeddings_if_missing(smith=smith, dataset_name=dataset, client=client)

    first_ex = next(smith.client.list_examples(dataset_name=dataset))
    user_message = first_ex.inputs.get("user_message")
    index_ex = next(smith.client.list_examples(dataset_name=index, metadata={"user_message": user_message}))
    index_emb = np.array(index_ex.outputs.get("embedding")).reshape(1, -1)

    for example in smith.client.list_examples(dataset_name=dataset):
        user_message = example.inputs.get("user_message")
        if example.outputs.get("embedding"):
            q_emb = example.outputs["embedding"]
            q_emb = np.array(q_emb).reshape(1, -1)
        else:
            continue
        sims = cosine_similarity(index_emb, q_emb)
        outputs = example.outputs
        outputs["similarity"] = sims.flatten()[0]
        smith.update_example(example_id=example.id, outputs=outputs)

    list_sims = []
    for example in smith.client.list_examples(dataset_name=dataset):
        sim = example.outputs.get("similarity")
        if sim is not None and isinstance(sim, (float, int)):
            list_sims.append((sim, example))

    ranking = sorted(list_sims, key=lambda x: x[0], reverse=True)
    return ranking[:top_k]

    # refs = idx["embeddings"]
    # meta = list(idx["meta"])
    # if refs.size == 0:
    #     return []

    # client = EmbeddingClient()
    # q_emb = client.embed_texts([query])[0]

    # from sklearn.metrics.pairwise import cosine_similarity

    # sims = cosine_similarity(refs, q_emb.reshape(1, -1)).reshape(-1)
    # order = np.argsort(-sims)[:top_k]
    # results = []
    # for i in order:
    #     results.append({"score": float(sims[i]), "meta": meta[i]})
    # return results
