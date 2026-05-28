"""
Download a clean text corpus for next-word prediction training.

Defaults to WikiText-103:
    python download_dataset.py

Smaller example:
    python download_dataset.py --dataset ag_news
"""

import argparse
import os

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: datasets. Install it with `python -m pip install datasets`."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Download a Hugging Face text dataset.")
    parser.add_argument("--dataset", default="Salesforce/wikitext", help="Dataset name.")
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config. Defaults to wikitext-103-v1 for WikiText.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to download.")
    parser.add_argument("--text-column", default="text", help="Column containing text.")
    parser.add_argument("--output", default=os.path.join("data", "corpus.txt"))
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Optional max lines to write.")
    return parser.parse_args()


def normalize_text(value):
    text = str(value).strip()
    return " ".join(text.split())


def main():
    args = parse_args()
    config = args.config

    dataset_name = args.dataset
    if dataset_name == "wikitext":
        dataset_name = "Salesforce/wikitext"

    if dataset_name == "Salesforce/wikitext" and config is None:
        config = "wikitext-103-v1"

    dataset_args = [dataset_name]
    if config:
        dataset_args.append(config)

    print(f"Downloading {dataset_name}" + (f"/{config}" if config else "") + f" ({args.split})...")
    dataset = load_dataset(*dataset_args, split=args.split)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    written = 0
    skipped = 0
    with open(args.output, "w", encoding="utf-8") as output_file:
        for item in dataset:
            if args.text_column not in item:
                available = ", ".join(item.keys())
                raise KeyError(
                    f"Column {args.text_column!r} was not found. Available columns: {available}"
                )

            text = normalize_text(item[args.text_column])
            if len(text) < args.min_chars:
                skipped += 1
                continue

            output_file.write(text + "\n")
            written += 1

            if args.limit and written >= args.limit:
                break

    print(f"Done. Wrote {written:,} lines to {args.output}.")
    print(f"Skipped {skipped:,} short or empty lines.")


if __name__ == "__main__":
    main()
