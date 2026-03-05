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


def load_all_stats(folder: str) -> list[dict[str, Any]]:
    """Load per-conversation average tokens and latency from examples.xlsx."""
    root = Path(folder)
    if not root.is_dir():
        return []

    stats: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        xlsx = child / "examples.xlsx"
        if not xlsx.is_file():
            continue
        df = pd.read_excel(xlsx)
        if df is None or df.empty:
            continue
        row: dict[str, Any] = {"conversation_name": child.name}
        for col in ["time_seconds", "total_tokens", "prompt_tokens", "completion_tokens"]:
            if col in df.columns:
                row[f"avg_{col}"] = round(float(df[col].dropna().mean()), 2)
            else:
                row[f"avg_{col}"] = np.nan
        stats.append(row)
    return stats


def _best_run_id(rank_result: dict) -> int | None:
    """Return the new_id of the top-1 response most similar to the golden best."""
    for _bid, top_list in rank_result.get("best_response_top_k", {}).items():
        if top_list:
            return top_list[0]["new_id"]
    return None


def build_summary_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a per-conversation summary DataFrame with an overall (mean) row.

    Each best response (rating=5) is expanded into its own row with a
    ``best_id`` column identifying which best it is.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        best_sims = r.get("best_avg_similarity", {})
        best_top_k = r.get("best_response_top_k", {})

        if best_sims:
            for bid in sorted(best_sims, key=lambda x: int(x) if x.isdigit() else x):
                top_list = best_top_k.get(bid, [])
                top1_id: Any = ""
                top1_score = np.nan
                if top_list:
                    top1_id = top_list[0]["new_id"]
                    top1_score = top_list[0]["score"]
                rows.append(
                    {
                        "conversation_name": r["conversation_name"],
                        "best_id": bid,
                        "best_top1_new_id": top1_id,
                        "best_top1_score": top1_score,
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
                    "best_top1_new_id": "",
                    "best_top1_score": np.nan,
                    "average_similarity": r["average_similarity"],
                    "avg_vs_avg_similarity": r["avg_vs_avg_similarity"],
                    "best_avg_similarity": np.nan,
                }
            )

    df = pd.DataFrame(rows)

    score_cols = ["average_similarity", "avg_vs_avg_similarity",
                  "best_avg_similarity", "best_top1_score"]
    overall: dict[str, Any] = {
        "conversation_name": "OVERALL",
        "best_id": "",
        "best_top1_new_id": "",
    }
    for col in score_cols:
        vals = df[col].dropna()
        overall[col] = round(float(vals.mean()), 6) if len(vals) > 0 else np.nan
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
    return df


def build_follow_up_df(
    results: list[dict[str, Any]],
    fu_results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a per-conversation follow-up summary DataFrame with OVERALL row."""
    rank_by_name = {r["conversation_name"]: r for r in results}
    fu_by_name = {fu["conversation_name"]: fu for fu in fu_results}

    rows: list[dict[str, Any]] = []
    for name, fu in fu_by_name.items():
        per_row = fu.get("per_row", {})

        run_sims: dict[str, list[float]] = {}
        run_maxes: list[float] = []
        run_avgs: list[float] = []
        for rid in sorted(per_row, key=lambda x: int(x) if x.isdigit() else x):
            rdata = per_row[rid]
            sims = [m["similarity"] for m in rdata.get("best_matches", [])]
            run_sims[f"run{rid}"] = [round(s, 4) for s in sims]
            if sims:
                run_maxes.append(max(sims))
                run_avgs.append(float(np.mean(sims)))

        score = round(float(np.mean(run_maxes)), 6) if run_maxes else np.nan
        avg_score = round(float(np.mean(run_avgs)), 6) if run_avgs else np.nan

        ok_count = len(fu.get("ok_follow_ups", []))

        best_rid = _best_run_id(rank_by_name.get(name, {}))
        best_run_sims_str = ""
        best_run_fu_avg = np.nan
        if best_rid is not None:
            rdata = per_row.get(str(best_rid), {})
            sims = [m["similarity"] for m in rdata.get("best_matches", [])]
            if sims:
                best_run_sims_str = str([round(s, 4) for s in sims])
                best_run_fu_avg = round(float(np.mean(sims)), 6)

        rows.append({
            "conversation_name": name,
            "n_ok_follow_ups": ok_count,
            "per_run_max_sims": str(run_sims),
            "follow_up_score": score,
            "follow_up_avg_similarity": avg_score,
            "best_run_id": best_rid if best_rid is not None else "",
            "best_run_fu_sims": best_run_sims_str,
            "best_run_fu_avg": best_run_fu_avg,
        })

    df = pd.DataFrame(rows)

    score_cols = ["follow_up_score", "follow_up_avg_similarity", "best_run_fu_avg"]
    overall: dict[str, Any] = {
        "conversation_name": "OVERALL",
        "n_ok_follow_ups": "",
        "per_run_max_sims": "",
        "best_run_id": "",
        "best_run_fu_sims": "",
    }
    for col in score_cols:
        vals = df[col].dropna()
        overall[col] = round(float(vals.mean()), 6) if len(vals) > 0 else np.nan
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
    return df


def load_all_embeddings_variance(folder: str) -> list[dict[str, Any]]:
    """Compute total variance (mean squared distance from centroid) per conversation.

    Reads ``embeddings.csv`` from each immediate sub-folder of *folder*.
    Returns one dict per conversation with keys
    ``conversation_name`` and ``total_variance``.
    """
    root = Path(folder)
    if not root.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        csv_path = child / "embeddings.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path, index_col=0)
        if df.empty:
            continue
        vectors = df.values.astype(float)
        centroid = vectors.mean(axis=0)
        total_var = float(np.mean(np.sum((vectors - centroid) ** 2, axis=1)))
        results.append({
            "conversation_name": child.name,
            "total_variance": round(total_var, 6),
        })
    return results


def build_variance_df(variances: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a per-conversation variance DataFrame with an OVERALL (mean) row."""
    df = pd.DataFrame(variances)
    if df.empty:
        return df

    mean_var = float(df["total_variance"].dropna().mean())
    overall = {
        "conversation_name": "OVERALL",
        "total_variance": round(mean_var, 6),
    }
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
    return df


_STATS_NUM_COLS = [
    "avg_time_seconds", "avg_total_tokens",
    "avg_prompt_tokens", "avg_completion_tokens",
]


def build_stats_df(stats: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a per-conversation stats DataFrame with OVERALL row."""
    df = pd.DataFrame(stats)
    present = [c for c in _STATS_NUM_COLS if c in df.columns and df[c].notna().any()]

    overall: dict[str, Any] = {"conversation_name": "OVERALL"}
    for col in present:
        vals = df[col].dropna()
        overall[col] = round(float(vals.mean()), 2) if len(vals) > 0 else np.nan
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
    return df
