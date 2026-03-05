from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from notebooks.utils.loader import (
    build_follow_up_df,
    build_stats_df,
    build_summary_df,
    build_variance_df,
    load_all_embeddings_variance,
    load_all_follow_up_results,
    load_all_results,
    load_all_stats,
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
    "best_top1_score",
    "average_similarity",
    "avg_vs_avg_similarity",
    "best_avg_similarity",
]
SCORE_LABELS = {
    "best_top1_score": "Best Top1 Score",
    "average_similarity": "Score2: Avg Similarity",
    "avg_vs_avg_similarity": "Score4: AvgVec vs AvgVec",
    "best_avg_similarity": "Score1: Best Avg Similarity",
}

FOLLOW_UP_SCORE_COLS = [
    "follow_up_score",
    "follow_up_avg_similarity",
    "best_run_fu_avg",
]

_STATS_NUM_COLS = [
    "avg_time_seconds",
    "avg_total_tokens",
    "avg_prompt_tokens",
    "avg_completion_tokens",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


_CSS_GREEN = "background-color: #c6efce; color: #006100"
_CSS_YELLOW = "background-color: #ffeb9c; color: #9c5700"
_CSS_RED = "background-color: #ffc7ce; color: #9c0006"
_CSS_GRAY = "background-color: #f2f2f2; color: #808080"


def _build_diff_df(
    target: pd.DataFrame,
    baseline: pd.DataFrame,
    score_cols: list[str],
    join_keys: list[str],
) -> pd.DataFrame:
    """Build a DataFrame of diff values (target - baseline) per score column."""
    base_lookup: dict[tuple, dict[str, float]] = {}
    for _, row in baseline.iterrows():
        key = tuple(row[k] for k in join_keys)
        base_lookup[key] = {c: row[c] for c in score_cols if c in row.index}

    rows: list[dict] = []
    for _, row in target.iterrows():
        out: dict = {k: row[k] for k in join_keys}
        key = tuple(row[k] for k in join_keys)
        base = base_lookup.get(key)
        for col in score_cols:
            t_val = row.get(col)
            if base is None:
                out[col] = np.nan
                continue
            b_val = base.get(col)
            if pd.isna(t_val) or b_val is None or pd.isna(b_val):
                out[col] = np.nan
            else:
                out[col] = float(t_val) - float(b_val)
        rows.append(out)
    return pd.DataFrame(rows)


def _style_diff(
    diff_df: pd.DataFrame,
    score_cols: list[str],
    higher_is_better: bool = True,
    upper: float = 0.05,
    lower: float = -0.05,
) -> pd.io.formats.style.Styler:
    """Apply 4-tier discrete coloring to a diff DataFrame.

    For higher_is_better=True:
      diff >= upper  → green  (significantly improved)
      lower <= diff < upper → yellow (near-zero / slight change)
      diff < lower   → red   (significantly regressed)

    For higher_is_better=False (lower is better):
      diff <= -upper → green  (significantly improved)
      -upper < diff < abs(lower) → yellow
      diff >= abs(lower) → red (significantly regressed)
    """
    present = [c for c in score_cols if c in diff_df.columns]
    abs_lower = abs(lower)

    def _color(val: float) -> str:
        if pd.isna(val):
            return _CSS_GRAY
        if higher_is_better:
            if val >= upper:
                return _CSS_GREEN
            if val <= lower:
                return _CSS_RED
            return _CSS_YELLOW
        else:
            if val <= -upper:
                return _CSS_GREEN
            if val >= abs_lower:
                return _CSS_RED
            return _CSS_YELLOW

    return diff_df.style.map(_color, subset=present)


# ---------------------------------------------------------------------------
# show_* — single-dataset display
# ---------------------------------------------------------------------------


def show_scores(
    data_dir: str,
    label: str = "Dataset",
    columns: list[str] | None = None,
) -> None:
    """Load results from a single directory and display summary table + bar chart.

    *columns* – list of column names to display. ``None`` shows all.
    """
    results = load_all_results(data_dir)
    df = build_summary_df(results)
    if columns is not None:
        keep = [c for c in df.columns if c in columns or c in ("conversation_name", "best_id")]
        df = df[keep]

    display(Markdown(f"## {label}"))
    display(Markdown(f"`{data_dir}` — {len(results)} conversations"))
    _show_table(df, title=f"{label} Scores")
    _plot_scores(df, title=f"{label} Score Distribution")


def show_follow_up_scores(
    data_dir: str,
    label: str = "Dataset",
    columns: list[str] | None = None,
) -> None:
    """Display per-run follow-up max similarities and the avg-of-run-max score.

    *columns* – list of column names to display. ``None`` shows all.
    """
    results = load_all_results(data_dir)
    fu_results = load_all_follow_up_results(data_dir)

    if not fu_results:
        display(Markdown("**Follow-up results not found.** Run `rank` first."))
        return

    display(Markdown(f"## {label} — Follow-up Question Similarity"))
    display(Markdown(f"`{data_dir}` — {len(fu_results)} conversations"))

    df = build_follow_up_df(results, fu_results)
    if columns is not None:
        keep = [c for c in df.columns if c in columns or c == "conversation_name"]
        df = df[keep]

    score_cols = [c for c in FOLLOW_UP_SCORE_COLS if c in df.columns]
    styler = (
        df.style.set_caption(f"{label} Follow-up Scores")
        .background_gradient(subset=score_cols, cmap="RdYlGn", vmin=0, vmax=1)
        .format({c: "{:.4f}" for c in score_cols})
        .set_properties(**{"text-align": "center"}, subset=score_cols)
    )
    display(styler)


def show_stats(
    data_dir: str,
    label: str = "Dataset",
    columns: list[str] | None = None,
) -> None:
    """Display average tokens and latency per conversation.

    *columns* – list of column names to display. ``None`` shows all.
    """
    stats = load_all_stats(data_dir)
    if not stats:
        display(Markdown("**Stats not found.** No examples.xlsx files found."))
        return

    display(Markdown(f"## {label} — Token & Latency Stats"))
    display(Markdown(f"`{data_dir}` — {len(stats)} conversations"))

    df = build_stats_df(stats)
    if columns is not None:
        keep = [c for c in df.columns if c in columns or c == "conversation_name"]
        df = df[keep]

    present = [c for c in _STATS_NUM_COLS if c in df.columns and df[c].notna().any()]

    fmt: dict[str, str] = {}
    for c in present:
        fmt[c] = "{:.2f}" if "time" in c else "{:.0f}"

    styler = (
        df.style.set_caption(f"{label} Avg Token & Latency")
        .format(fmt)
        .set_properties(**{"text-align": "center"}, subset=present)
    )
    display(styler)


def show_variance(
    data_dir: str,
    label: str = "Dataset",
) -> None:
    """Display per-conversation total variance of embedding vectors.

    Total variance = mean squared Euclidean distance from centroid.
    The OVERALL row shows the mean of per-conversation variances.
    """
    variances = load_all_embeddings_variance(data_dir)
    if not variances:
        display(Markdown("**Variance not available.** No embeddings.csv files found."))
        return

    display(Markdown(f"## {label} — Embedding Vector Variance"))
    display(Markdown(f"`{data_dir}` — {len(variances)} conversations"))

    df = build_variance_df(variances)

    styler = (
        df.style.set_caption(f"{label} Total Variance (from centroid)")
        .format({"total_variance": "{:.6f}"})
        .set_properties(**{"text-align": "center"}, subset=["total_variance"])
    )
    display(styler)


# ---------------------------------------------------------------------------
# compare_* — two-dataset comparison (target colored by diff from baseline)
# ---------------------------------------------------------------------------


def compare_scores(
    baseline_dir: str,
    target_dir: str,
    baseline_label: str = "Baseline",
    target_label: str = "Target",
    upper: float = 0.0,
    lower: float = -0.05,
    columns: list[str] | None = None,
) -> None:
    """Show diff table of agent-response scores (target - baseline).

    *columns* – score column names to include. ``None`` uses all ``SCORE_COLS``.
    """
    base_df = build_summary_df(load_all_results(baseline_dir))
    target_df = build_summary_df(load_all_results(target_dir))

    cols = [c for c in (columns or SCORE_COLS) if c in target_df.columns]
    join_keys = ["conversation_name", "best_id"]
    diff_df = _build_diff_df(target_df, base_df, cols, join_keys)

    display(Markdown(f"## {target_label} vs {baseline_label} — Agent Response Diff"))
    display(
        Markdown(
            f"Target: `{target_dir}` / Baseline: `{baseline_dir}`  \n"
            f"Green (>= {upper:+g}) / Yellow ({lower:+g} 〜 {upper:+g}) "
            f"/ Red (< {lower:+g}) / Gray (N/A)"
        )
    )

    styler = _style_diff(diff_df, cols, higher_is_better=True, upper=upper, lower=lower)
    styler.set_caption(f"{target_label} − {baseline_label}: Agent Response")
    styler.format({c: "{:+.4f}" for c in cols})
    styler.set_properties(**{"text-align": "center"}, subset=cols)
    display(styler)


def compare_follow_up_scores(
    baseline_dir: str,
    target_dir: str,
    baseline_label: str = "Baseline",
    target_label: str = "Target",
    upper: float = 0.05,
    lower: float = -0.05,
    columns: list[str] | None = None,
) -> None:
    """Show diff table of follow-up scores (target - baseline).

    *columns* – score column names to include. ``None`` uses all ``FOLLOW_UP_SCORE_COLS``.
    """
    target_results = load_all_results(target_dir)
    target_fu = load_all_follow_up_results(target_dir)
    if not target_fu:
        display(Markdown("**Follow-up results not found for target.**"))
        return

    base_fu = load_all_follow_up_results(baseline_dir)
    base_df = (
        build_follow_up_df(load_all_results(baseline_dir), base_fu)
        if base_fu
        else pd.DataFrame()
    )
    target_df = build_follow_up_df(target_results, target_fu)

    cols = [c for c in (columns or FOLLOW_UP_SCORE_COLS) if c in target_df.columns]
    join_keys = ["conversation_name"]
    diff_df = _build_diff_df(target_df, base_df, cols, join_keys)

    display(Markdown(f"## {target_label} vs {baseline_label} — Follow-up Diff"))
    display(
        Markdown(
            f"Target: `{target_dir}` / Baseline: `{baseline_dir}`  \n"
            f"Green (>= {upper:+g}) / Yellow ({lower:+g} 〜 {upper:+g}) "
            f"/ Red (< {lower:+g}) / Gray (N/A)"
        )
    )

    styler = _style_diff(
        diff_df, cols, higher_is_better=True, upper=upper, lower=lower,
    )
    styler.set_caption(f"{target_label} − {baseline_label}: Follow-up")
    styler.format({c: "{:+.4f}" for c in cols})
    styler.set_properties(**{"text-align": "center"}, subset=cols)
    display(styler)


def compare_stats(
    baseline_dir: str,
    target_dir: str,
    baseline_label: str = "Baseline",
    target_label: str = "Target",
    upper: float = 5.0,
    lower: float = -5.0,
    columns: list[str] | None = None,
) -> None:
    """Show diff table of stats (target - baseline). Lower is better.

    *columns* – stat column names to include. ``None`` uses all ``_STATS_NUM_COLS``.
    """
    target_stats = load_all_stats(target_dir)
    if not target_stats:
        display(Markdown("**Stats not found for target.**"))
        return

    base_stats = load_all_stats(baseline_dir)
    base_df = build_stats_df(base_stats) if base_stats else pd.DataFrame()
    target_df = build_stats_df(target_stats)

    candidates = columns or _STATS_NUM_COLS
    present = [
        c for c in candidates
        if c in target_df.columns and target_df[c].notna().any()
    ]
    join_keys = ["conversation_name"]
    diff_df = _build_diff_df(target_df, base_df, present, join_keys)

    display(Markdown(f"## {target_label} vs {baseline_label} — Token & Latency Diff"))
    display(
        Markdown(
            f"Target: `{target_dir}` / Baseline: `{baseline_dir}`  \n"
            f"Green (<= {-upper:+g}) / Yellow ({-upper:+g} 〜 {abs(lower):+g}) "
            f"/ Red (>= {abs(lower):+g}) / Gray (N/A)"
        )
    )

    styler = _style_diff(
        diff_df, present, higher_is_better=False, upper=upper, lower=lower,
    )
    styler.set_caption(f"{target_label} − {baseline_label}: Stats")

    fmt: dict[str, str] = {}
    for c in present:
        fmt[c] = "{:+.2f}" if "time" in c else "{:+.0f}"
    styler.format(fmt)
    styler.set_properties(**{"text-align": "center"}, subset=present)
    display(styler)


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


def show_similarity_heatmap(folder: str, conversation_name: str) -> None:
    """Display a heatmap of the similarity matrix for one conversation."""
    csv_path = (
        Path(folder) / conversation_name / "rank_results" / "similarity_matrix.csv"
    )
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
