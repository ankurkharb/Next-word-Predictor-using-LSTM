"""
Build a mixed WikiText + conversational corpus.

This keeps the broad language knowledge from WikiText but boosts dialogue
patterns so prompts like "hello how are" can produce conversational continuations.

Usage:
    python build_mixed_corpus.py
"""

import argparse
import os
import random
import re
from collections import defaultdict

from datasets import load_dataset


WIKITEXT_PATH = os.path.join("data", "corpus.txt")
OUTPUT_PATH = os.path.join("data", "mixed_corpus.txt")
CONVERSATION_DATASET = "pixelsandpointers/better_daily_dialog"


COMMON_DIALOGUE_LINES = [
    "hello how are you",
    "hello how are you doing",
    "hello how are you doing today",
    "hi how are you",
    "hi how are you doing",
    "hey how are you",
    "hey how are you doing",
    "how are you",
    "how are you doing",
    "how are you today",
    "how are things going",
    "i am fine thank you",
    "i am doing well thank you",
    "nice to meet you",
    "good morning how are you",
    "good evening how are you",
]


def normalize_text(text):
    text = str(text).replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_wikitext_lines(path, max_lines, seed):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing data/corpus.txt. Run `python download_dataset.py` first.")

    lines = []
    with open(path, "r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            text = normalize_text(line)
            if text:
                lines.append(text)

    random.Random(seed).shuffle(lines)
    return lines[:max_lines]


def load_dialogue_lines(repeat):
    rows = []
    for split in ["train", "validation", "test"]:
        dataset = load_dataset(CONVERSATION_DATASET, split=split)
        rows.extend(dataset)

    dialogue_turns = defaultdict(list)
    utterances = []
    for row in rows:
        utterance = normalize_text(row["utterance"])
        if not utterance:
            continue

        utterances.append(utterance)
        dialogue_turns[row["dialog_id"]].append(utterance)

    dialogue_lines = []
    for turns in dialogue_turns.values():
        if turns:
            dialogue_lines.append(" ".join(turns))

    boosted = []
    for _ in range(repeat):
        boosted.extend(utterances)
        boosted.extend(dialogue_lines)

    for _ in range(max(50, repeat * 50)):
        boosted.extend(COMMON_DIALOGUE_LINES)

    return boosted


def parse_args():
    parser = argparse.ArgumentParser(description="Build a mixed training corpus.")
    parser.add_argument("--wikitext-path", default=WIKITEXT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--wiki-lines", type=int, default=180000)
    parser.add_argument("--dialogue-repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print("Loading WikiText sample...")
    wiki_lines = load_wikitext_lines(args.wikitext_path, args.wiki_lines, args.seed)
    print(f"WikiText lines: {len(wiki_lines):,}")

    print(f"Loading conversational dataset: {CONVERSATION_DATASET}...")
    dialogue_lines = load_dialogue_lines(args.dialogue_repeat)
    print(f"Boosted dialogue lines: {len(dialogue_lines):,}")

    all_lines = wiki_lines + dialogue_lines
    random.shuffle(all_lines)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as output_file:
        for line in all_lines:
            output_file.write(line + "\n")

    print(f"Done. Wrote {len(all_lines):,} lines to {args.output}.")


if __name__ == "__main__":
    main()
