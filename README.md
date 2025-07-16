# Next-word-Predictor-using-LSTM
This project demonstrates how to build a text generation model using an LSTM-based neural network. 
The model learns the context and structure of the input text and generates coherent next-word predictions based on a given input phrase.

## ✅ Features

- Converts raw FAQ text into n-gram training sequences.
- Trains an LSTM model to predict the next word in a sequence.
- Supports generating new text based on a custom seed phrase.
- Demonstrates key NLP steps: tokenization, padding, sequence modeling, and generation.
- Uses TensorFlow/Keras for model building and training.

## 🔧 Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy

## 📌 How It Works

1. **Tokenization**: Splits text into word-level tokens and assigns each a unique integer.
2. **Sequence Generation**: Creates sequences of increasing length from each line of text.
3. **Padding**: Ensures all sequences are the same length using pre-padding.
4. **Model Architecture**: 
   - Embedding Layer
   - Two stacked LSTM layers
   - Dense output layer with softmax activation
5. **Training**: Uses categorical cross-entropy loss to learn next-word prediction.
6. **Text Generation**: Given a seed phrase, the model predicts one word at a time and appends it to grow the text.
