import os
import numpy as np
from typing import List, Dict, Any, Optional

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

        # Try 0: openai.OpenAI
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

        # Try 1: langchain_openai.OpenAIEmbeddings
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

        # Try 2: langchain_openai.AzureOpenAIEmbeddings
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