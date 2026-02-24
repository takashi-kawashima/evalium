import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# load local.env for CLI use
load_dotenv("local.env")

from evaluator import build_reference_embeddings, rank_query


def build_index_cmd(args):
    res = build_reference_embeddings(args.data_dir, args.out, rating_threshold=args.threshold)
    print(json.dumps(res, ensure_ascii=False))


def rank_cmd(args):
    results = rank_query(args.embeddings, args.query, top_k=args.top_k)
    for r in results:
        print(f"score={r['score']:.4f}\tfolder={r['meta']['folder']}\trating={r['meta']['rating']}\n{r['meta']['response']}\n---")


def main():
    parser = argparse.ArgumentParser(description="Evalium CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build-index")
    p_build.add_argument("--data-dir", required=True)
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--threshold", type=float, default=4.0)
    p_build.add_argument("--assume-all", action="store_true", help="Treat all rows as reference responses (no rating required)")

    p_rank = sub.add_parser("rank")
    p_rank.add_argument("--embeddings", required=True)
    p_rank.add_argument("--query", required=True)
    p_rank.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    if args.cmd == "build-index":
        build_index_cmd(args)
    elif args.cmd == "rank":
        rank_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
