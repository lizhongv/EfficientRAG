import argparse
import os
import pickle
import sys

if True:
    pro_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(pro_dir)
    os.chdir(pro_dir)
    from src.retrievers.utils.utils import load_passages
    from src.retrievers.embeddings import Embedder, ModelTypes
    from src.log.logging_config import logger, LBLUE, LGREEN, LRED, RESET
    logger.info(f"pro_dir: {pro_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--passages",
        type=str,
        required=True,
        help="Path to passages (tsv or jsonl file)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output file")
    parser.add_argument("--model_type", type=str, default="e5-base-v2", choices=list(ModelTypes.keys()))
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=int(2e6), help="passages per chunk")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    return args


def main(opts):
    embedder = Embedder(
        opts.model_type,
        opts.model_name_or_path,
        batch_size=opts.batch_size,
        chunk_size=opts.chunk_size,
        text_normalize=True,
    )
    logger.info(f"Loading passages from {opts.passages}")
    data = load_passages(opts.passages)
    output_dir = opts.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for idx, (ids, embeddings) in embedder.embed_passages(data):
        output_file = os.path.join(output_dir, f"passages_{idx:02d}")
        with open(output_file, "wb") as f:
            pickle.dump((ids, embeddings), f)

        logger.info(f"Save {len(ids)} embeddings to {output_file}")


if __name__ == "__main__":
    options = parse_args()
    main(options)
