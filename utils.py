import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd

class SentimentEnsemble:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load models from HuggingFace
        self.models = []
        self.tokenizers = []

        model_paths = [
            "CHIPP-AI/model1",
            "CHIPP-AI/model2",
            "CHIPP-AI/model3"
        ]

        for path in model_paths:
            model, tokenizer = self.load_model(path)
            self.models.append(model)
            self.tokenizers.append(tokenizer)

    def load_model(self, repo_id):
        print(f"Loading model from {repo_id}")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        model = model.to(self.device)
        model.eval()
        return model, tokenizer

    def predict(self, text):
        predictions = []

        # Get predictions from each model
        with torch.no_grad():
            for model, tokenizer in zip(self.models, self.tokenizers):
                inputs = tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                predictions.append(probs.cpu().numpy())

        # Ensemble predictions (average probabilities)
        ensemble_pred = np.mean(predictions, axis=0)
        final_pred = np.argmax(ensemble_pred)

        # Map prediction to label
        label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = label_mapping[final_pred]

        # Get confidence scores
        confidence_scores = {
            "Negative": float(ensemble_pred[0][0]),
            "Neutral": float(ensemble_pred[0][1]),
            "Positive": float(ensemble_pred[0][2])
        }

        return sentiment, confidence_scores

def predict_sentiment(text):
    if not text.strip():
        return "Please enter some text", {}

    sentiment, confidence = model.predict(text)

    # Format confidence scores as percentages
    confidence = {k: f"{v*100:.2f}%" for k, v in confidence.items()}

    return sentiment, confidence

def predict_sentiment_csv(file_obj, column_name):
    """
    Expects a CSV file and a column name. If the user provides a column name that exists in the CSV,
    that column is used for text. Otherwise, the first column is used.
    Each non-empty row is passed through predict_sentiment(), and a count of each
    predicted sentiment is returned.
    """
    try:
        # gr.File with type="file" returns a file path.
        df = pd.read_csv(file_obj)
    except Exception as e:
        return {"error": f"Error reading CSV file: {e}"}

    if len(df) > 100:
      df = df.head(100)
    # Use the provided column name if it exists; otherwise, default to the first column.
    if column_name and column_name in df.columns:
        text_col = column_name
    else:
        text_col = df.columns[0]

    counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
    
    # Iterate over the rows in the chosen column
    for text in df[text_col]:
        if not isinstance(text, str):
            text = str(text)
        if not text.strip():
            continue  # Skip empty strings
        
        sentiment, _ = predict_sentiment(text)
        if sentiment == "Please enter some text":
            continue
        counts[sentiment] += 1

    return counts
