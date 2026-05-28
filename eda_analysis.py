"""
EDA for the WikiText corpus used by the next-word predictor.

Run after downloading the corpus:
    python eda_analysis.py
"""

import os
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.util import ngrams
from wordcloud import WordCloud


CORPUS_PATH = os.path.join("data", "corpus.txt")
OUTPUT_DIR = "eda_output"


def save_plot(filename, facecolor=None):
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=facecolor)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError("Missing data/corpus.txt. Run `python download_dataset.py` first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    with open(CORPUS_PATH, "r", encoding="utf-8") as corpus_file:
        raw_text = corpus_file.read()

    sentences = [s.strip() for s in raw_text.split("\n") if len(s.strip()) > 20]
    words_raw = re.findall(r"\b[a-zA-Z]+\b", raw_text.lower())
    stop_words = set(stopwords.words("english"))
    words_clean = [word for word in words_raw if word not in stop_words]

    word_freq = Counter(words_raw)
    clean_word_freq = Counter(words_clean)
    sent_lengths = [len(sentence.split()) for sentence in sentences]

    summary = {
        "Total characters": len(raw_text),
        "Total sentences": len(sentences),
        "Total words": len(words_raw),
        "Unique words": len(set(words_raw)),
        "Vocabulary clean": len(set(words_clean)),
    }

    for label, value in summary.items():
        print(f"{label:<18}: {value:,}")

    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "corpus_summary.csv"), index=False)

    # Plot 1: top word frequencies.
    top_50 = word_freq.most_common(50)
    words_list, counts = zip(*top_50)

    plt.figure(figsize=(16, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 50))
    plt.bar(words_list, counts, color=colors)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.title("Top 50 Most Frequent Words", fontsize=16, fontweight="bold")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    save_plot("01_word_frequency.png")

    # Plot 2: Zipf's law.
    all_counts = sorted(word_freq.values(), reverse=True)[:5000]
    ranks = np.arange(1, len(all_counts) + 1)

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, all_counts, "b-", linewidth=2, label="Actual distribution")
    zipf_ideal = all_counts[0] / ranks
    plt.loglog(ranks, zipf_ideal, "r--", linewidth=2, label="Zipf's Law ideal")
    plt.title("Zipf's Law - Word Frequency vs Rank", fontsize=16, fontweight="bold")
    plt.xlabel("Rank (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("02_zipfs_law.png")

    # Plot 3: sentence length distribution.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(sent_lengths, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    plt.title("Sentence Length Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Words per Sentence")
    plt.ylabel("Count")
    plt.axvline(
        np.mean(sent_lengths),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(sent_lengths):.1f}",
    )
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot(
        sent_lengths,
        vert=True,
        patch_artist=True,
        boxprops={"facecolor": "steelblue", "alpha": 0.7},
    )
    plt.title("Sentence Length Boxplot", fontsize=14, fontweight="bold")
    plt.ylabel("Words per Sentence")
    save_plot("03_sentence_length.png")

    # Plot 4: vocabulary coverage curve.
    sorted_counts = sorted(word_freq.values(), reverse=True)
    total_words = sum(sorted_counts)
    cumulative = np.cumsum(sorted_counts) / total_words * 100
    vocab_sizes = np.arange(1, len(cumulative) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(vocab_sizes[:20000], cumulative[:20000], "g-", linewidth=2)
    for pct in [50, 80, 90, 95]:
        idx = int(np.searchsorted(cumulative, pct))
        plt.axhline(y=pct, color="gray", linestyle=":", alpha=0.7)
        plt.axvline(x=idx, color="gray", linestyle=":", alpha=0.7)
        plt.annotate(
            f"{pct}% covered\nby {idx:,} words",
            xy=(idx, pct),
            xytext=(idx + 500, pct - 8),
            fontsize=8,
            color="darkred",
        )
    plt.title("Vocabulary Coverage Curve", fontsize=16, fontweight="bold")
    plt.xlabel("Vocabulary Size (unique words)")
    plt.ylabel("% of Total Text Covered")
    plt.grid(True, alpha=0.3)
    save_plot("04_vocab_coverage.png")

    # Plot 5: bigram and trigram analysis.
    top_bigrams = Counter(ngrams(words_clean, 2)).most_common(20)
    top_trigrams = Counter(ngrams(words_clean, 3)).most_common(15)

    bg_labels = [f"{a} {b}" for (a, b), _ in top_bigrams]
    bg_counts = [count for _, count in top_bigrams]
    tg_labels = [f"{a} {b} {c}" for (a, b, c), _ in top_trigrams]
    tg_counts = [count for _, count in top_trigrams]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.barh(bg_labels[::-1], bg_counts[::-1], color="coral")
    ax1.set_title("Top 20 Bigrams", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Frequency")

    ax2.barh(tg_labels[::-1], tg_counts[::-1], color="mediumpurple")
    ax2.set_title("Top 15 Trigrams", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Frequency")
    save_plot("05_ngrams.png")

    # Plot 6: word cloud.
    wordcloud = WordCloud(
        width=1400,
        height=700,
        background_color="black",
        colormap="plasma",
        max_words=300,
        min_font_size=10,
    ).generate_from_frequencies(clean_word_freq)

    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(
        "Word Cloud (stopwords removed)",
        fontsize=18,
        fontweight="bold",
        color="white",
        pad=20,
    )
    save_plot("06_wordcloud.png", facecolor="black")

    # Plot 7: corpus statistics dashboard.
    stats = {
        "Total Words": len(words_raw),
        "Unique Words": len(set(words_raw)),
        "Total Sentences": len(sentences),
        "Avg Sent Length": round(float(np.mean(sent_lengths)), 1),
        "Vocabulary clean": len(set(words_clean)),
        "Top Word Count": word_freq.most_common(1)[0][1],
    }

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("Corpus Statistics Dashboard", fontsize=18, fontweight="bold")
    grid = gridspec.GridSpec(2, 3, figure=fig)
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    for index, ((label, value), color) in enumerate(zip(stats.items(), colors)):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        ax.set_facecolor(color)
        display_value = f"{value:,}" if isinstance(value, int) else str(value)
        ax.text(
            0.5,
            0.6,
            display_value,
            ha="center",
            va="center",
            fontsize=26,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.25,
            label,
            ha="center",
            va="center",
            fontsize=13,
            transform=ax.transAxes,
            color="#333",
        )
        ax.axis("off")

    save_plot("07_stats_dashboard.png")


if __name__ == "__main__":
    main()
