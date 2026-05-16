"""
LSTM Next-Word Predictor — Flask Web Application
=================================================
A production-ready Flask API + frontend for next-word prediction.

Usage:
    python app.py          # Development server (port 5000)
    gunicorn app:app       # Production server

Endpoints:
    GET  /              → Serves the frontend UI
    POST /api/predict   → Returns next-word predictions
    GET  /api/health    → Health check
    GET  /api/metadata  → Model metadata
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# LOAD MODEL AT STARTUP (once, not per-request)
# ─────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

print("[*] Loading model artifacts...")

# Load the trained LSTM model
model = load_model(os.path.join(MODEL_DIR, 'lstm_model.keras'))
print("  > Model loaded")

# Load the tokenizer
with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)
print("  > Tokenizer loaded")

# Load metadata
with open(os.path.join(MODEL_DIR, 'metadata.json'), 'r') as f:
    metadata = json.load(f)
print("  > Metadata loaded")

max_len = metadata['max_len']
vocab_size = metadata['vocab_size']

# Build reverse word index for fast lookup
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

print(f"  > Ready! Vocab: {vocab_size} words, Max seq len: {max_len}")
print("=" * 50)


# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
def get_top_predictions(text, top_k=5, temperature=0.8):
    """Get top-K next word predictions with probabilities."""
    token_text = tokenizer.texts_to_sequences([text])[0]

    if len(token_text) == 0:
        return [], text

    padded = pad_sequences([token_text], maxlen=max_len - 1, padding='pre')
    predictions = model.predict(padded, verbose=0)[0]

    # Apply temperature scaling
    predictions = np.log(predictions + 1e-10) / temperature
    predictions = np.exp(predictions) / np.sum(np.exp(predictions))

    # Get top-K indices
    top_indices = np.argsort(predictions)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        word = reverse_word_index.get(idx, None)
        if word and idx != 0:  # Skip padding token
            results.append({
                'word': word,
                'probability': round(float(predictions[idx]) * 100, 2)
            })

    return results


def predict_sequence(text, num_words=5, temperature=0.8):
    """Predict a sequence of N next words."""
    result = text
    word_sequence = []

    for _ in range(num_words):
        token_text = tokenizer.texts_to_sequences([result])[0]

        if len(token_text) == 0:
            break

        padded = pad_sequences([token_text], maxlen=max_len - 1, padding='pre')
        predictions = model.predict(padded, verbose=0)[0]

        # Temperature sampling
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        # Sample from top candidates
        top_indices = np.argsort(predictions)[-10:]
        top_probs = predictions[top_indices]
        top_probs = top_probs / top_probs.sum()

        predicted_index = np.random.choice(top_indices, p=top_probs)
        predicted_word = reverse_word_index.get(predicted_index, None)

        if predicted_word:
            result += " " + predicted_word
            word_sequence.append(predicted_word)
        else:
            break

    return result, word_sequence


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the frontend UI."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict next words.
    
    Request body:
        {
            "text": "what is the",
            "num_words": 5,
            "temperature": 0.8
        }
    
    Response:
        {
            "input_text": "what is the",
            "top_predictions": [
                {"word": "course", "probability": 45.2},
                {"word": "total", "probability": 22.1},
                ...
            ],
            "generated_text": "what is the course fee for data",
            "generated_words": ["course", "fee", "for", "data", "science"]
        }
    """
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request body'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        num_words = min(int(data.get('num_words', 5)), 20)  # Cap at 20
        temperature = max(0.1, min(float(data.get('temperature', 0.8)), 2.0))  # Clamp 0.1–2.0

        # Get top predictions for the immediate next word
        top_predictions = get_top_predictions(text, top_k=5, temperature=temperature)

        # Generate a full sequence
        generated_text, generated_words = predict_sequence(text, num_words, temperature)

        return jsonify({
            'input_text': text,
            'top_predictions': top_predictions,
            'generated_text': generated_text,
            'generated_words': generated_words,
            'settings': {
                'temperature': temperature,
                'num_words': num_words
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vocab_size': vocab_size
    })


@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Return model metadata."""
    return jsonify(metadata)


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
