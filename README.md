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

### 4. Train the model

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
$env:MAX_TRAINING_SEQUENCES="100000"; $env:EPOCHS="10"; python train_model.py
```

### 5. Plot training history

```bash
python plot_training.py
```

This creates `eda_output/08_training_curves.png`.

### 6. Copy EDA charts to the web app

```powershell
New-Item -ItemType Directory -Force -Path static\eda
Copy-Item eda_output\*.png static\eda -Force
```

### 7. Run the app

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

- Corpus lines: `798,784`
- Raw words: `82,254,211`
- Vocabulary limit: `10,000`
- Sequence length: `20`
- Training sequences: `225,000`
- Validation sequences: `25,000`
- Epochs trained: `7`
- Final train accuracy: `0.1818`
- Final validation accuracy: `0.1643`

## Interview Summary

I built an end-to-end NLP pipeline starting with corpus acquisition, then did
EDA including Zipf's law verification, n-gram analysis, vocabulary coverage,
and word cloud visualization. I trained a Bidirectional LSTM with early
stopping and learning-rate scheduling on WikiText-103, then deployed it as a
Flask API with a real-time prediction UI and an EDA dashboard.

## Project Structure

```text
app.py                    # Flask backend
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
