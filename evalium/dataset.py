from dataclasses import dataclass, field
from os import path
import pandas as pd
from typing import Any, Dict, List, Optional
import os
import json
import glob
import numpy as np

@dataclass
class Dataset:
    name: str
    description: Optional[str] = None
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str):
        self.df.to_excel(path, index=False)
        self.save_embeddings(os.path.join(os.path.dirname(path), "embeddings.csv"))

    def save_embeddings(self, path: str):
        emb_df = pd.DataFrame({'index':self.embeddings.keys()})
        emb_2d = np.array(list(self.embeddings.values()))
        emb_df = pd.concat([emb_df, pd.DataFrame(emb_2d)], axis=1)
        emb_df.to_csv(path, index=False)
    
    def load_embeddings(self, folder: str):
        emb_df = load_embeddings(folder)
        if emb_df is not None:
            self.embeddings = {row['index']: row.drop('index').values for _, row in emb_df.iterrows()}

    @classmethod
    def from_examples(cls, examples, dataset_name: str):
        df = pd.DataFrame([{**example['inputs'], **example['outputs']} for example in examples])
        input_keys = list(examples[0]['inputs'].keys()) if examples else []
        output_keys = list(examples[0]['outputs'].keys()) if examples else []
        metadata = examples[0]['metadata'] if examples else {}
        dataset = cls(
            name=dataset_name,
            description="",
            df=df,
            input_keys=input_keys,
            output_keys=output_keys,
            metadata=metadata
        )
        return dataset

    @classmethod
    def from_folder(cls, folder: str):
        dataset_name = os.path.basename(folder)
        input_json = load_input_json(folder)
        df = load_df(folder)
        if df is None:
            df = pd.DataFrame()
        # detect likely columns
        text_col = "agent_response"
        rating_col = "rating"
        if rating_col not in df.columns:
            df[rating_col] = None
        df['user_message'] = input_json['user_message']
        dataset = cls(
            name=dataset_name,
            description="",
            df=df,
            input_keys=["user_message"],
            output_keys=[text_col,"run_index","follow_up_questions","tools_and_arguments","iteration_count","time_seconds","total_tokens","prompt_tokens","completion_tokens",rating_col],
            metadata=input_json
        )
        dataset.load_embeddings(os.path.join(folder))
        return dataset

@dataclass
class Example:
    dataset_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

def load_embeddings(folder: str) -> Optional[np.ndarray]:
    files = glob.glob(os.path.join(folder, "embeddings.csv"))
    emb = files[0] if files else None
    if emb:
        df = pd.read_csv(emb)
        return df
    return None

def load_input_json(folder: str) -> dict:
    p = os.path.join(folder, "input.json")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_df(folder: str) -> Optional[pd.DataFrame]:
    files = glob.glob(os.path.join(folder, "dataset_with_emb.xlsx"))
    emb = files[0] if files else None
    if emb:
        df = pd.read_excel(emb)
    else:
        files = glob.glob(os.path.join(folder, "*.xlsx"))
        excel = files[0] if files else None
        df = pd.read_excel(excel) if excel else None
    return df
