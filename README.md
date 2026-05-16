# NeuraType — LSTM Next-Word Predictor

An AI-powered next-word prediction web app using a Bidirectional LSTM neural network, served via Flask with a premium dark-themed UI.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Flask](https://img.shields.io/badge/Flask-3.1-green)

## Features

- **Optimized LSTM** — Bidirectional layers, Dropout, BatchNorm, learning rate scheduling
- **Temperature sampling** — Control prediction creativity (focused ↔ creative)
- **Top-5 predictions** — See the most likely next words with probability bars
- **Beautiful UI** — Dark glassmorphism theme with smooth animations
- **One-click deploy** — Ready for Render, Railway, or Docker

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This creates `models/lstm_model.keras`, `models/tokenizer.pkl`, and `models/metadata.json`.

### 3. Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Deploy**

## Deploy with Docker

```bash
docker build -t neuratype .
docker run -p 5000:5000 neuratype
```

## API

### `POST /api/predict`

```json
{
  "text": "what is the",
  "num_words": 5,
  "temperature": 0.8
}
```

**Response:**
```json
{
  "input_text": "what is the",
  "top_predictions": [
    {"word": "course", "probability": 45.2},
    {"word": "total", "probability": 22.1}
  ],
  "generated_text": "what is the course fee for data science",
  "generated_words": ["course", "fee", "for", "data", "science"]
}
```

## Project Structure

```
├── app.py               # Flask backend
├── train_model.py        # Model training script
├── requirements.txt      # Dependencies
├── Procfile              # Render/Heroku
├── Dockerfile            # Docker deployment
├── models/               # Trained model artifacts
├── templates/index.html  # Frontend
├── static/
│   ├── style.css         # Dark theme
│   └── app.js            # Frontend logic
└── lstm_next_word_predictor.ipynb  # Original notebook
```

## License

MIT
