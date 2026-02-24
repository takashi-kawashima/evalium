import os
import json
import pandas as pd
import hashlib
from datetime import datetime
from typing import Dict, Any
from langsmith import Client

class LangSmithIntegration:
    client:Client
    def __init__(self):
        key = os.getenv("LANGSMITH_API_KEY")
        if not key:
            return
        # try likely client names; this is defensive and best-effort
        try:
            import langsmith as _ls
            Client = getattr(_ls, "Client", None) or getattr(_ls, "LangSmithClient", None)
        except Exception:
            Client = None

        if Client is None:
            return
        self.client = Client(api_key=key)

    def create_dataset(
            self,
            dataset_name: str, 
            description: str, 
            df:pd.DataFrame,
            input_keys: list, 
            output_keys: list) -> bool:
        try:
            self.client.upload_dataframe(
                df=df,
                input_keys=input_keys,
                output_keys=output_keys,
                name=dataset_name,
                description=description
            )

        except Exception as e:
            print("Error sending to LangSmith:", e)
            return False
        
    def list_examples(self, dataset_name: str):
        try:
            datasets = self.client.list_examples(dataset_name=dataset_name)
            return datasets
        except Exception as e:
            print("Error listing datasets from LangSmith:", e)
            return []
    
    def update_example(self, example_id: str, outputs: Dict[str, Any]) -> bool:
        try:
            self.client.update_example(example_id=example_id, outputs=outputs)
            return True
        except Exception as e:
            print("Error updating example in LangSmith:", e)
            return False


    def try_send_to_langsmith(self,metadata: Dict[str, Any],df:pd.DataFrame, artifact_path: str) -> bool:
        """Attempt to send a minimal run record to LangSmith if client is available.

        This is best-effort: if the `langsmith` client isn't installed or an error
        occurs, the function will return False and not raise.
        """
        try:
            run = self.client.create_run(name=f"{metadata['config_name']}_shop{metadata['shop_id']}_{metadata['prompt_date']}" , run_type="llm" , inputs=metadata)

            self.client.update_run(run_id = run.id, run_type="llm" ,outputs={"artifact_path":artifact_path})

        except Exception as e:
            print("Error sending to LangSmith:", e)
            return False

        return False

    def _sha1_of_obj(self,obj: Any) -> str:
        raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:10]


    def save_metadata(self,metadata: Dict[str, Any], out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        key = self._sha1_of_obj(metadata)
        fname = os.path.join(out_dir, f"metadata-{key}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return fname