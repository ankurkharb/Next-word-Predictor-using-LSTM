"""
Train the LSTM next-word predictor on a real text corpus.

Usage:
    python download_dataset.py
    python train_model.py

Optional environment variables:
    CORPUS_PATH=data/corpus.txt
    MAX_VOCAB_SIZE=10000
    SEQ_LENGTH=20
    MAX_TRAINING_SEQUENCES=250000
    VALIDATION_SPLIT=0.1
    SHUFFLE_SEED=42
    EPOCHS=50
    BATCH_SIZE=128
    MODEL_DIR=models
"""

import json
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


MODEL_DIR = os.environ.get("MODEL_DIR", "models")
CORPUS_PATH = os.environ.get("CORPUS_PATH", os.path.join("data", "corpus.txt"))

MAX_VOCAB_SIZE = int(os.environ.get("MAX_VOCAB_SIZE", "10000"))
SEQ_LENGTH = int(os.environ.get("SEQ_LENGTH", os.environ.get("MAX_SEQUENCE_LEN", "20")))
MAX_TRAINING_SEQUENCES = int(os.environ.get("MAX_TRAINING_SEQUENCES", "250000"))
VALIDATION_SPLIT = float(os.environ.get("VALIDATION_SPLIT", "0.1"))
EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "128"))
LSTM_UNITS = int(os.environ.get("LSTM_UNITS", "200"))
MIN_LINE_CHARS = int(os.environ.get("MIN_LINE_CHARS", "50"))
SHUFFLE_SEED = int(os.environ.get("SHUFFLE_SEED", "42"))


def load_corpus(path, min_line_chars):
    """Load non-empty corpus lines from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus not found at {path!r}. Run `python download_dataset.py` first."
        )

    lines = []
    with open(path, "r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            text = line.strip()
            if len(text) >= min_line_chars:
                lines.append(text)

    if not lines:
        raise ValueError(f"No usable text lines found in {path!r}.")

    return lines


def count_training_sequences(lines, tokenizer, max_sequences):
    """Count generated next-word examples, respecting the optional training cap."""
    count = 0
    for line in lines:
        tokenized_line = tokenizer.texts_to_sequences([line])[0]
        if len(tokenized_line) < 2:
            continue

        count += len(tokenized_line) - 1
        if max_sequences and count >= max_sequences:
            return max_sequences

    return count


def sequence_generator(lines, tokenizer, seq_length, max_sequences):
    """Yield padded prefix windows and sparse next-word labels."""
    yielded = 0

    for line in lines:
        tokenized_line = tokenizer.texts_to_sequences([line])[0]
        if len(tokenized_line) < 2:
            continue

        for end_index in range(1, len(tokenized_line)):
            sequence = tokenized_line[max(0, end_index - seq_length): end_index + 1]
            x_tokens = sequence[:-1]
            y_token = sequence[-1]

            x = np.zeros(seq_length, dtype=np.int32)
            x[-len(x_tokens):] = x_tokens

            yielded += 1
            yield x, np.int32(y_token)

            if max_sequences and yielded >= max_sequences:
                return


def build_dataset(lines, tokenizer, seq_length, max_sequences):
    output_signature = (
        tf.TensorSpec(shape=(seq_length,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(lines, tokenizer, seq_length, max_sequences),
        output_signature=output_signature,
    )
    return dataset.shuffle(10000).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(vocab_size, seq_length):
    model = Sequential(
        [
            tf.keras.Input(shape=(seq_length,)),
            Embedding(vocab_size, EMBEDDING_DIM),
            Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
            Dropout(0.1),
            LSTM(LSTM_UNITS),
            Dropout(0.1),
            Dense(128, activation="relu"),
            Dense(vocab_size, activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def predict_next_words(model, tokenizer, text, seq_length, num_words=5, temperature=0.8):
    """Predict a small sample sequence after training."""
    reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
    result = text

    for _ in range(num_words):
        token_text = tokenizer.texts_to_sequences([result])[0]
        if not token_text:
            break

        padded = pad_sequences([token_text], maxlen=seq_length, padding="pre")
        predictions = model.predict(padded, verbose=0)[0]

        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        top_indices = np.argsort(predictions)[-10:]
        top_probs = predictions[top_indices]
        top_probs = top_probs / top_probs.sum()

        predicted_index = int(np.random.choice(top_indices, p=top_probs))
        predicted_word = reverse_word_index.get(predicted_index)
        if not predicted_word:
            break

        result += " " + predicted_word

    return result


def main():
    print("=" * 60)
    print("  LSTM Next-Word Predictor - Training")
    print("=" * 60)

    print(f"\nLoading corpus from {CORPUS_PATH}...")
    lines = load_corpus(CORPUS_PATH, MIN_LINE_CHARS)
    print(f"Loaded {len(lines):,} corpus lines")

    np.random.default_rng(SHUFFLE_SEED).shuffle(lines)

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(lines)

    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    split_index = int(len(lines) * (1 - VALIDATION_SPLIT))
    train_lines = lines[:split_index]
    val_lines = lines[split_index:]

    train_sequence_limit = int(MAX_TRAINING_SEQUENCES * (1 - VALIDATION_SPLIT))
    val_sequence_limit = MAX_TRAINING_SEQUENCES - train_sequence_limit
    train_sequences = count_training_sequences(train_lines, tokenizer, train_sequence_limit)
    val_sequences = count_training_sequences(val_lines, tokenizer, val_sequence_limit)
    total_sequences = train_sequences + val_sequences

    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Sequence length: {SEQ_LENGTH}")
    print(f"Training sequences: {train_sequences:,}")
    print(f"Validation sequences: {val_sequences:,}")
    steps_per_epoch = max(1, int(np.ceil(train_sequences / BATCH_SIZE)))
    validation_steps = max(1, int(np.ceil(val_sequences / BATCH_SIZE)))
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Validation steps: {validation_steps:,}")

    training_data = build_dataset(
        lines=train_lines,
        tokenizer=tokenizer,
        seq_length=SEQ_LENGTH,
        max_sequences=train_sequence_limit,
    )
    validation_data = build_dataset(
        lines=val_lines,
        tokenizer=tokenizer,
        seq_length=SEQ_LENGTH,
        max_sequences=val_sequence_limit,
    )

    print("\nBuilding model...")
    model = build_model(vocab_size=vocab_size, seq_length=SEQ_LENGTH)
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, "lstm_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nTraining started...\n")
    history = model.fit(
        training_data,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nExporting model artifacts...")

    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)
    print(f"Saved {os.path.join(MODEL_DIR, 'tokenizer.pkl')}")

    metadata = {
        "corpus_path": CORPUS_PATH,
        "line_count": len(lines),
        "vocab_size": vocab_size,
        "max_vocab_size": MAX_VOCAB_SIZE,
        "seq_length": SEQ_LENGTH,
        "max_len": SEQ_LENGTH + 1,
        "embedding_dim": EMBEDDING_DIM,
        "total_sequences": total_sequences,
        "training_sequences": train_sequences,
        "validation_sequences": val_sequences,
        "epochs_trained": len(history.history["loss"]),
        "final_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }

    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(f"Saved {os.path.join(MODEL_DIR, 'metadata.json')}")

    with open(os.path.join(MODEL_DIR, "training_history.json"), "w", encoding="utf-8") as history_file:
        json.dump(history.history, history_file)
    print(f"Saved {os.path.join(MODEL_DIR, 'training_history.json')}")

    print("\nQuick prediction test:")
    for phrase in ["the united states", "in the early", "according to"]:
        prediction = predict_next_words(model, tokenizer, phrase, SEQ_LENGTH)
        print(f'  Input: "{phrase}"')
        print(f'  Output: "{prediction}"\n')

    print("=" * 60)
    print("Training complete. Run `python app.py` to start the web server.")
    print("=" * 60)


if __name__ == "__main__":
    main()
