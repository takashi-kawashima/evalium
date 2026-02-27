import argparse

from dotenv import load_dotenv

from evalium.evaluator import build_index, rank_query

# load local.env for CLI use
load_dotenv("local.env")


def build_index_cmd(args):
    index_conv = build_index(args.data_dir, rating_threshold=args.threshold)
    print(f"Index built, dataset id: {index_conv.name}")


def rank_cmd(args):
    results = rank_query(args.index, args.dataset, top_k=args.top_k)
    print("Ranking results:")
    for i, res in enumerate(results):
        print(
            f"Rank {i + 1}: id={res[1].name} , score={res[0]:.4f} , user_message ={res[1].user_message} "
        )


def main():
    parser = argparse.ArgumentParser(description="Evalium CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build-index")
    p_build.add_argument("--data-dir", required=True)
    p_build.add_argument("--threshold", type=float, default=4.0)

    p_rank = sub.add_parser("rank")
    p_rank.add_argument("--index", required=True)
    p_rank.add_argument("--dataset", required=True)
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
