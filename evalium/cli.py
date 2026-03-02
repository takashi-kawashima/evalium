import argparse

from dotenv import load_dotenv

from evalium.evaluator import build_index, rank_query

# load local.env for CLI use
load_dotenv("local.env")


def build_index_cmd(args):
    force = getattr(args, "force", False)
    if force:
        print("Force rebuild: clearing existing embeddings")
    index_conv = build_index(args.data_dir, rating_threshold=args.threshold, force=force)
    print(f"Index built, dataset id: {index_conv.name}")


def rank_cmd(args):
    force = getattr(args, "force", False)
    if force:
        print("Force rebuild: clearing existing embeddings for dataset")
    results = rank_query(args.index, args.dataset, top_k=args.top_k, force=force)

    print(f"\n=== Rank Results: {results['conversation_name']} ===\n")

    print(f"[Score 2] Average similarity (full matrix): {results['average_similarity']:.4f}")
    print(f"[Score 4] Avg vector vs avg vector:          {results['avg_vs_avg_similarity']:.4f}\n")

    print(f"[Score 1] Best (rating=5) response top-{args.top_k}:")
    for best_id, top_list in results["best_response_top_k"].items():
        avg = results["best_avg_similarity"][best_id]
        print(f"  Golden best id={best_id} (avg similarity to all new: {avg:.4f}):")
        for rank, entry in enumerate(top_list, 1):
            print(f"    Rank {rank}: new_id={entry['new_id']}, score={entry['score']:.4f}")
    print()

    print(f"[Score 3] Average vector vs new responses (top-{args.top_k}):")
    for rank, entry in enumerate(results["average_vector_ranking"][:args.top_k], 1):
        print(f"  Rank {rank}: new_id={entry['new_id']}, score={entry['score']:.4f}")

    output_dir = results.get("_output_dir", args.dataset + "/rank_results")
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evalium CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build-index")
    p_build.add_argument("--data-dir", required=True)
    p_build.add_argument("--threshold", type=float, default=4.0)
    p_build.add_argument("--force", action="store_true", default=False,
                         help="Force rebuild: clear existing embeddings and regenerate all")

    p_rank = sub.add_parser("rank")
    p_rank.add_argument("--index", required=True)
    p_rank.add_argument("--dataset", required=True)
    p_rank.add_argument("--top-k", type=int, default=5)
    p_rank.add_argument("--force", action="store_true", default=False,
                        help="Force rebuild: clear existing embeddings and regenerate all")

    args = parser.parse_args()
    if args.cmd == "build-index":
        build_index_cmd(args)
    elif args.cmd == "rank":
        rank_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
