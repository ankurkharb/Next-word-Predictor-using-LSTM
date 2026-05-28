# NeuraType - LSTM Next-Word Predictor

An end-to-end NLP project for next-word prediction with corpus acquisition,
EDA, Bidirectional LSTM training, Flask APIs, and a web UI.

## Features

- WikiText-103 corpus downloader using Hugging Face `datasets`
- EDA portfolio dashboard with Zipf's law, vocabulary coverage, n-grams, word cloud, and training curves
- Bidirectional LSTM with top-10k vocabulary, sparse labels, validation monitoring, early stopping, and LR scheduling
- Flask prediction API with metadata-driven padding
- `/analysis` page for showing all EDA charts

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the corpus

```bash
python download_dataset.py
```

This creates `data/corpus.txt` from WikiText-103 (`Salesforce/wikitext`).

For a smaller dataset:

```bash
python download_dataset.py --dataset ag_news
```

### 3. Run EDA

```bash
python eda_analysis.py
```

This creates plots in `eda_output/`.

### 4. Build the mixed corpus

```bash
python build_mixed_corpus.py
```

This creates `data/mixed_corpus.txt` by mixing WikiText with boosted
DailyDialog-style conversational text.

### 5. Train the model

```bash
python train_model.py
```

This creates:

- `models/lstm_model.keras`
- `models/tokenizer.pkl`
- `models/metadata.json`
- `models/training_history.json`

Useful PowerShell knobs:

```powershell
$env:CORPUS_PATH="data/mixed_corpus.txt"; $env:MAX_TRAINING_SEQUENCES="600000"; python train_model.py
```

### 6. Plot training history

```bash
python plot_training.py
```

This creates `eda_output/08_training_curves.png`.

### 7. Copy EDA charts to the web app

```powershell
New-Item -ItemType Directory -Force -Path static\eda
Copy-Item eda_output\*.png static\eda -Force
```

### 8. Run the app

```bash
python app.py
```

Open:

- Predictor: `http://localhost:5000`
- Analysis dashboard: `http://localhost:5000/analysis`

## API

### `POST /api/predict`

```json
{
  "text": "the united states",
  "num_words": 5,
  "temperature": 0.8
}
```

### `GET /api/health`

Returns model status, vocabulary size, and sequence length.

### `GET /api/metadata`

Returns the saved training metadata.

## Current Training Run

- Corpus: WikiText-103 + `pixelsandpointers/better_daily_dialog`
- Corpus lines: `1,099,176`
- Vocabulary limit: `10,000`
- Sequence length: `20`
- Training sequences: `540,000`
- Validation sequences: `60,000`
- Epochs trained: `8`
- Final train accuracy: `0.2003`
- Final validation accuracy: `0.1840`
- Example: `hello how are` predicts `you`

## Interview Summary

I built an end-to-end NLP pipeline starting with corpus acquisition, then did
EDA including Zipf's law verification, n-gram analysis, vocabulary coverage,
and word cloud visualization. I trained a Bidirectional LSTM with early
stopping and learning-rate scheduling on WikiText-103, then deployed it as a
Flask API with a real-time prediction UI and an EDA dashboard.

## Project Structure

```text
app.py                    # Flask backend
build_mixed_corpus.py     # WikiText + conversation corpus builder
download_dataset.py       # Corpus downloader
eda_analysis.py           # EDA script for plots 1-7
plot_training.py          # Training history plot
train_model.py            # Model training script
requirements.txt          # Dependencies
data/corpus.txt           # Downloaded corpus, ignored by git
eda_output/               # Generated EDA outputs
models/                   # Trained model artifacts
static/eda/               # EDA charts served by Flask
templates/index.html      # Predictor UI
templates/analysis.html   # EDA dashboard UI
```

## Deploy

Render/Railway/Docker can serve the trained model with:

```bash
gunicorn app:app --timeout 120
```

## License

MIT
