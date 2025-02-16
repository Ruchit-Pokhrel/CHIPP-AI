import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

        # Get confidence scores (as float values)
        confidence_scores = {
            "Negative": float(ensemble_pred[0][0]),
            "Neutral": float(ensemble_pred[0][1]),
            "Positive": float(ensemble_pred[0][2])
        }

        return sentiment, confidence_scores

def predict_sentiment(text):
    if not text.strip():
        return "Please enter some text", None

    sentiment, confidence = model.predict(text)
    # Multiply by 100 for percentage values
    confidence_numeric = {k: v * 100 for k, v in confidence.items()}

    labels = list(confidence_numeric.keys())
    sizes = list(confidence_numeric.values())
    
    fig, ax = plt.subplots()
    # Create pie chart without autopct
    wedges, texts = ax.pie(sizes, labels=labels, startangle=90)
    ax.axis('equal')  # Ensure pie is a circle.
    ax.set_title("Confidence Scores (%)")
    
    # Compute percentages and add legend outside the pie-chart
    total = sum(sizes)
    legend_labels = [f"{label}: {(size/total)*100:.1f}%" for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    return sentiment, fig

def predict_sentiment_csv(file_obj, column_name):
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        return {"error": f"Error reading CSV file: {e}"}
    
    # Process only the first 100 rows
    if len(df) > 10:
        df = df.head(10)
    
    # Use the provided column name if it exists; otherwise, default to the first column.
    text_col = column_name if column_name and column_name in df.columns else df.columns[0]

    counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
    
    # Analyze each row in the selected column
    for text in df[text_col]:
        if not isinstance(text, str):
            text = str(text)
        if not text.strip():
            continue
        
        sentiment, _ = predict_sentiment(text)
        if sentiment == "Please enter some text":
            continue
        counts[sentiment] += 1

    labels = list(counts.keys())
    sizes = list(counts.values())
    
    fig, ax = plt.subplots()
    # Create pie chart without autopct
    wedges, texts = ax.pie(sizes, labels=labels, startangle=90)
    ax.axis('equal')
    ax.set_title("Sentiment Distribution (First 100 Rows)")
    
    total = sum(sizes)
    legend_labels = [f"{label}: {(size/total)*100:.1f}%" if total > 0 else f"{label}: 0.0%" 
                     for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="Sentiments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    return fig
