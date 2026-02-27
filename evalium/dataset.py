from dataclasses import dataclass, field
from os import path
import pandas as pd
from typing import Any, Dict, List, Optional
import os
import json
import glob
import numpy as np


@dataclass
class Conversation:
    name: str
    turn: str
    path: Optional[str] = None
    description: Optional[str] = None
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, Any] = field(default_factory=dict)

    def apply_master_info(self,master:pd.DataFrame):
        shopid = self.metadata['shop_id']
        topic = self.metadata['user_message']
        rows = master.query(f'shop_id == {shopid} and topic == "{topic}" and conversation_step == 1')
        self.df['rating'] = None
        self.df['case'] = None
        for _, row in rows.iterrows():
            case = row['case']
            best = row['best_response_id']
            oks = row['ok_response_id'].split(",") if pd.notna(row['ok_response_id']) else []
            follow_up_qs = row['ok_follow_up_list'].split(",") if pd.notna(row['ok_follow_up_list']) else []
            for ok in oks:
                self.df.at[int(ok) - 1, 'rating'] = 3.0
            self.df.at[best - 1, 'rating'] = 5.0
            self.df.at[best - 1, 'case'] = case
    
    def save(self):
        excel_path = os.path.join(self.path, "examples.xlsx") if self.path else None
        self.df.to_excel(excel_path, index=False)
        self.save_embeddings()

    def save_embeddings(self):
        emb_df = pd.DataFrame({'index':self.embeddings.keys()})
        emb_2d = np.array(list(self.embeddings.values()))
        emb_df = pd.concat([emb_df, pd.DataFrame(emb_2d)], axis=1)
        path = os.path.join(os.path.dirname(self.path), "embeddings.csv")
        emb_df.to_csv(path, index=False)
    
    def load_embeddings(self, folder: str):
        emb_df = load_embeddings(folder)
        if emb_df is not None:
            self.embeddings = {row['index']: row.drop('index').values for _, row in emb_df.iterrows()}

    def fetch_dataset_embeddings(self):
        embeddings = []
        for i, example in self.df.iterrows():
            if self.embeddings.get(i) is not None:
                emb = self.embeddings[i]
                embeddings.append(emb)
        embeddings = np.array(embeddings, dtype=np.float32)
        # modify all_emb is average vector of collection of all_emb, and normalize to unit vector below:
        if len(embeddings) > 0:
            ave_embeddings = np.mean(embeddings, axis=0, keepdims=False)
        else:
            ave_embeddings = None # Assuming embedding size of 384; adjust as needed
        return embeddings, ave_embeddings

    @classmethod
    def from_examples(cls, examples, path: str, dataset_name: str, turn:str):
        df = pd.DataFrame([{**example['inputs'], **example['outputs']} for example in examples])
        input_keys = list(examples[0]['inputs'].keys()) if examples else []
        output_keys = list(examples[0]['outputs'].keys()) if examples else []
        embeddings = examples[0]['embeddings'] if examples else {}
        metadata = examples[0]['metadata'] if examples else {}
        dataset = cls(
            name=dataset_name,
            turn=turn,
            path=path,
            description="",
            df=df,
            input_keys=input_keys,
            output_keys=output_keys,
            embeddings=embeddings,
            metadata=metadata
        )
        return dataset

    @classmethod
    def from_path(cls, path:str, turn:str):
        dataset_name = os.path.basename(path)
        df = pd.read_excel(path)
        if df is None:
            df = pd.DataFrame()
        dataset = cls(
            name = dataset_name,
            turn = turn,
            path = path,
            description="",
            df=df,
            input_keys=["user_message"],
            output_keys=["embedding"],
            metadata={}
        )
        return dataset

    @classmethod
    def from_folder(cls, folder: str, turn:str):
        dataset_name = os.path.basename(folder)
        input_json = load_input_json(folder)
        df = cls.load_df(folder)
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
            turn=turn,
            path = folder,
            description="",
            df=df,
            input_keys=["user_message"],
            output_keys=[text_col,"run_index","follow_up_questions","tools_and_arguments","iteration_count","time_seconds","total_tokens","prompt_tokens","completion_tokens",rating_col],
            metadata=input_json
        )
        dataset.load_embeddings(os.path.join(folder))
        return dataset
    
    @classmethod
    def load_df(cls,folder: str) -> Optional[pd.DataFrame]:
        files = glob.glob(os.path.join(folder, "examples.xlsx"))
        emb = files[0] if files else None
        if emb:
            df = pd.read_excel(emb)
        else:
            files = glob.glob(os.path.join(folder, "*.xlsx"))
            excel = files[0] if files else None
            df = pd.read_excel(excel, index_col=0) if excel else None
        df.dropna(how='all', inplace=True)
        df['agent_response'] = df['agent_response'].astype(str)
        df['follow_up_questions'] = df['follow_up_questions'].astype(str)
        df['tools_and_arguments'] = df['tools_and_arguments'].astype(str)
        return df


@dataclass
class Conversations:
    name: str
    description: Optional[str] = None
    master: pd.DataFrame = field(default_factory=pd.DataFrame)
    conversations: Dict[str, Conversation] = field(default_factory=dict)


    @classmethod
    def from_folder(cls, folder: str):
        name = os.path.basename(folder)
        description = ""
        master_file = os.path.join(folder, "golden_data_master_table.xlsx")
        master = cls.load_master(master_file)
        conversations = {}
        for turn_path in glob.glob(os.path.join(folder, "*")):
            if os.path.isdir(turn_path):
                # retrieve turn name from folder
                turn = os.path.basename(turn_path)
                for examples_path in glob.glob(os.path.join(turn_path, "*")):
                    if os.path.isdir(examples_path):
                        exmaples_name = os.path.basename(examples_path)
                        conv = Conversation.from_folder(examples_path, turn=turn)
                        conversations[exmaples_name] = conv

        return cls(name=name, description=description, master=master, conversations=conversations)

    @classmethod
    def load_master(cls,master_file) -> pd.DataFrame:
        if not os.path.exists(master_file):
            print(f"Master table not found at {master_file}, please run the evaluation first to generate it.")
            return pd.DataFrame() # return empty dataframe if master not found
        master_df = pd.read_excel(master_file)
        return master_df

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
