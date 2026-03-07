#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --dataset <dir> [--index <dir>] [--top-k <n>] [--scenario]"
    echo ""
    echo "  --dataset   Parent directory containing dataset sub-folders (required)"
    echo "  --index     Golden dataset directory (default: data/goldendataset or data/goldendataset_scenario with --scenario)"
    echo "  --top-k     Number of top results (default: 5)"
    echo "  --scenario  Use scenario golden dataset (sets --index to data/goldendataset_scenario)"
    exit 1
}

INDEX="data/goldendataset"
DATASET=""
TOP_K=5
SCENARIO=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET="$2"; shift 2 ;;
        --index)    INDEX="$2";   shift 2 ;;
        --top-k)    TOP_K="$2";   shift 2 ;;
        --scenario) SCENARIO=1; shift ;;
        -h|--help)  usage ;;
        *)          echo "Unknown option: $1"; usage ;;
    esac
done

if [[ "$SCENARIO" -eq 1 ]]; then
    INDEX="data/goldendataset_scenario"
fi

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required"
    usage
fi

if [[ ! -d "$DATASET" ]]; then
    echo "Error: '$DATASET' is not a directory"
    exit 1
fi

found=0
for dir in "$DATASET"/*/; do
    [[ -d "$dir" ]] || continue
    dir="${dir%/}"
    found=1
    echo "=========================================="
    echo "Running rank: $dir"
    echo "=========================================="
    uv run python -m evalium.cli rank --index "$INDEX" --dataset "$dir" --top-k "$TOP_K"  --force
    echo ""
done

if [[ "$found" -eq 0 ]]; then
    echo "Error: No sub-directories found in '$DATASET'"
    exit 1
fi

echo "All done."
