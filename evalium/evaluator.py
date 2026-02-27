import os
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv

# from langsmith_integration import LangSmithIntegration
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
        if not resp_text:
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


def rank_query(
    index_path: str, dataset_folder: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    client = EmbeddingClient()
    # smith = LangSmithIntegration()
    index = Conversation.from_index(index_path)
    dataset = Conversation.from_folder(dataset_folder)
    add_embeddings(
        dataset=dataset, client=client, rating_threshold=-1.0
    )  # Add embeddings for all examples regardless of rating
    dataset.save()
    # retrieve conversataion key (conversation)
    conversation_name = dataset.metadata.get("name")
    index_emb = index.embeddings[conversation_name]
    # index_emb = np.array([float(s) for s in emb_str.strip('[]').split()])
    index_emb = np.array(index_emb).reshape(1, -1)
    for i, example in dataset.df.iterrows():
        user_message = example["user_message"]
        emb = dataset.embeddings.get(i)
        if emb is not None:
            q_emb = np.array(emb).reshape(1, -1)
        else:
            continue
        sims = cosine_similarity(index_emb, q_emb)
        dataset.df.at[i, "similarity"] = sims.flatten()[0]

    list_sims = []
    for i, example in dataset.df.iterrows():
        sim = example["similarity"]
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
