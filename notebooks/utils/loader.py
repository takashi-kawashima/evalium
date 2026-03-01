from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_all_results(folder: str) -> list[dict[str, Any]]:
    """Load rank_results.json from every immediate sub-folder of *folder*."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

    results: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        json_path = child / "rank_results" / "rank_results.json"
        if json_path.is_file():
            with open(json_path, encoding="utf-8") as f:
                results.append(json.load(f))
    if not results:
        raise FileNotFoundError(
            f"No rank_results.json found under sub-folders of {folder}"
        )
    return results


def load_all_follow_up_results(folder: str) -> list[dict[str, Any]]:
    """Load follow_up_results.json from every immediate sub-folder of *folder*."""
    root = Path(folder)
    if not root.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        json_path = child / "rank_results" / "follow_up_results.json"
        if json_path.is_file():
            with open(json_path, encoding="utf-8") as f:
                results.append(json.load(f))
    return results


def build_summary_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a per-conversation summary DataFrame with an overall (mean) row.

    Each best response (rating=5) is expanded into its own row with a
    ``best_id`` column identifying which best it is.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        best_sims = r.get("best_avg_similarity", {})

        if best_sims:
            for bid in sorted(best_sims, key=lambda x: int(x) if x.isdigit() else x):
                rows.append(
                    {
                        "conversation_name": r["conversation_name"],
                        "best_id": bid,
                        "average_similarity": r["average_similarity"],
                        "avg_vs_avg_similarity": r["avg_vs_avg_similarity"],
                        "best_avg_similarity": round(best_sims[bid], 6),
                    }
                )
        else:
            rows.append(
                {
                    "conversation_name": r["conversation_name"],
                    "best_id": "-",
                    "average_similarity": r["average_similarity"],
                    "avg_vs_avg_similarity": r["avg_vs_avg_similarity"],
                    "best_avg_similarity": np.nan,
                }
            )

    df = pd.DataFrame(rows)

    overall: dict[str, Any] = {
        "conversation_name": "OVERALL",
        "best_id": "",
        "average_similarity": round(df["average_similarity"].mean(), 6),
        "avg_vs_avg_similarity": round(df["avg_vs_avg_similarity"].mean(), 6),
        "best_avg_similarity": round(df["best_avg_similarity"].dropna().mean(), 6)
        if df["best_avg_similarity"].notna().any()
        else np.nan,
    }
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
    return df
