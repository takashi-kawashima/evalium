from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from notebooks.utils.loader import (
    build_summary_df,
    load_all_follow_up_results,
    load_all_results,
)

warnings.filterwarnings("ignore", message="Glyph .* missing from font")

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Hiragino Sans",
    "Hiragino Maru Gothic Pro",
    "Arial Unicode MS",
    "IPAexGothic",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]

SCORE_COLS = [
    "average_similarity",
    "avg_vs_avg_similarity",
    "best_avg_similarity",
]
SCORE_LABELS = {
    "average_similarity": "Score2: Avg Similarity",
    "avg_vs_avg_similarity": "Score4: AvgVec vs AvgVec",
    "best_avg_similarity": "Score1: Best Avg Similarity",
}


def _present_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in SCORE_COLS if c in df.columns and df[c].notna().any()]


def _style_scores(styler: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
    cols = _present_cols(styler.data)
    return (
        styler.background_gradient(subset=cols, cmap="RdYlGn", vmin=0, vmax=1)
        .format({c: "{:.4f}" for c in cols})
        .set_properties(**{"text-align": "center"}, subset=cols)
    )


def _show_table(df: pd.DataFrame, title: str) -> None:
    styler = df.style.set_caption(title).pipe(_style_scores)
    display(styler)


def _make_bar_label(row: pd.Series) -> str:
    name = row["conversation_name"]
    short = name[:18] + "…" if len(name) > 18 else name
    if "best_id" in row.index and row["best_id"]:
        return f"{short}\n(best={row['best_id']})"
    return short


def _plot_scores(df: pd.DataFrame, title: str) -> None:
    data = df[df["conversation_name"] != "OVERALL"]
    labels = data.apply(_make_bar_label, axis=1).values

    cols = _present_cols(df)
    if not cols:
        return

    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]

    x = np.arange(len(labels))

    for ax, col in zip(axes, cols):
        vals = data[col].values
        ax.bar(x, vals, color="#5b9bd5")
        ax.set_title(SCORE_LABELS.get(col, col))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)

        overall = df[df["conversation_name"] == "OVERALL"][col].values
        if len(overall) and not np.isnan(overall[0]):
            ax.axhline(y=overall[0], color="#c0392b", linestyle="--", linewidth=1)
            ax.text(
                len(labels) - 0.5,
                overall[0] + 0.02,
                f"OVERALL {overall[0]:.4f}",
                ha="right",
                fontsize=8,
                color="#c0392b",
            )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()


def show_scores(data_dir: str, label: str = "Dataset") -> None:
    """Load results from a single directory and display summary table + bar chart."""
    results = load_all_results(data_dir)
    df = build_summary_df(results)

    display(Markdown(f"## {label}"))
    display(Markdown(f"`{data_dir}` — {len(results)} conversations"))
    _show_table(df, title=f"{label} Scores")
    _plot_scores(df, title=f"{label} Score Distribution")


def _best_run_id(rank_result: dict) -> int | None:
    """Return the new_id of the top-1 response most similar to the golden best."""
    for _bid, top_list in rank_result.get("best_response_top_k", {}).items():
        if top_list:
            return top_list[0]["new_id"]
    return None


def show_follow_up_scores(data_dir: str, label: str = "Dataset") -> None:
    """Display per-run follow-up max similarities and the avg-of-run-max score."""
    results = load_all_results(data_dir)
    fu_results = load_all_follow_up_results(data_dir)

    if not fu_results:
        display(Markdown("**Follow-up results not found.** Run `rank` first."))
        return

    display(Markdown(f"## {label} — Follow-up Question Similarity"))
    display(Markdown(f"`{data_dir}` — {len(fu_results)} conversations"))

    rank_by_name = {r["conversation_name"]: r for r in results}
    fu_by_name = {fu["conversation_name"]: fu for fu in fu_results}

    rows: list[dict] = []
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
    overall: dict = {
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

    styler = (
        df.style
        .set_caption(f"{label} Follow-up Scores")
        .background_gradient(subset=score_cols, cmap="RdYlGn", vmin=0, vmax=1)
        .format({c: "{:.4f}" for c in score_cols})
        .set_properties(**{"text-align": "center"}, subset=score_cols)
    )
    display(styler)


def show_similarity_heatmap(folder: str, conversation_name: str) -> None:
    """Display a heatmap of the similarity matrix for one conversation."""
    csv_path = Path(folder) / conversation_name / "rank_results" / "similarity_matrix.csv"
    if not csv_path.is_file():
        print(f"File not found: {csv_path}")
        return

    sim = pd.read_csv(csv_path, index_col=0, encoding="utf-8_sig")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sim.columns)))
    ax.set_xticklabels(sim.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sim.index)))
    ax.set_yticklabels(sim.index, fontsize=7)
    ax.set_xlabel("New run index")
    ax.set_ylabel("Golden run index")
    ax.set_title(f"Similarity Matrix: {conversation_name[:40]}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    plt.show()
