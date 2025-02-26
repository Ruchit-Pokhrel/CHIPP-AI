import os
import json
import pandas as pd
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
from io import StringIO
from wordcloud import WordCloud




# Load environment variables and initialize Deepseek client
load_dotenv(dotenv_path='/content/drive/MyDrive/.env')
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables")
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

#############################################
# SENTIMENT ENSEMBLE CODE
#############################################
class SentimentEnsemble:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
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
        ensemble_pred = np.mean(predictions, axis=0)
        final_pred = np.argmax(ensemble_pred)
        label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = label_mapping[final_pred]


        confidence_scores = {
            "Negative": float(ensemble_pred[0][0]),
            "Neutral": float(ensemble_pred[0][1]),
            "Positive": float(ensemble_pred[0][2])
        }
        return sentiment, confidence_scores

# Instantiate the ensemble model
model = SentimentEnsemble()

def predict_sentiment(text):
    if not text.strip():
        return "Please enter some text", None

    sentiment, confidence = model.predict(text)
    confidence_numeric = {k: v * 100 for k, v in confidence.items()}
    labels = list(confidence_numeric.keys())
    sizes = list(confidence_numeric.values())

    fig, ax = plt.subplots()
    wedges, _ = ax.pie(sizes, labels=labels, startangle=90)
    ax.axis('equal')
    ax.set_title("Confidence Scores (%)")
    total = sum(sizes)
    legend_labels = [f"{label}: {(size/total)*100:.1f}%" for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    return sentiment, fig

def predict_sentiment_csv(file_obj, column_name):
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        return {"error": f"Error reading CSV file: {e}"}

    # Process only the first 10 rows (adjustable)
    if len(df) > 100:
        df = df.head(100)
    text_col = column_name if column_name and column_name in df.columns else df.columns[0]
    counts = {"Negative": 0, "Neutral": 0, "Positive": 0}


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
    wedges, _ = ax.pie(sizes, labels=labels, startangle=90)
    ax.axis('equal')
    ax.set_title("Sentiment Distribution (First 10 Rows)")
    total = sum(sizes)
    legend_labels = [f"{label}: {(size/total)*100:.1f}%" if total > 0 else f"{label}: 0.0%"
                     for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="Sentiments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    return fig

#############################################
# API KEYWORD EXTRACTION & WORD CLOUD CODE
#############################################
def process_input(input_data):
    """Convert input to a list of texts."""
    # If input is a DataFrame, use the "text" column if available, else the first column.
    if isinstance(input_data, pd.DataFrame):
        text_col = "text" if "text" in input_data.columns else input_data.columns[0]
        texts = input_data[text_col].fillna("").tolist()
        return texts
    # If input is a Series, convert to list.
    elif isinstance(input_data, pd.Series):
        return input_data.fillna("").tolist()
    # Otherwise, if it's a string.
    elif isinstance(input_data, str):
        return [input_data] if input_data.strip() else []
    else:
        return []

def clean_api_response(content):
    """Remove markdown code block formatting from API response."""
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content

def get_keywords_from_api(input_data):
    """Call API to extract keywords from reviews and return JSON."""
    system_prompt = """
As an AI text analyzer, your task is to extract key topics and sentiments from the provided reviews. Generate a JSON output with the following structure:
{
  "products": {
    "Product_A": {
      "positive_topics": {
        "topic1": frequency,
        "topic2": frequency
      },
      "negative_topics": {
        "topic1": frequency,
        "topic2": frequency
      }
    }
  }
}
Ensure your response contains ONLY the JSON object, with no additional text or explanation.
"""
    reviews = process_input(input_data)
    if not reviews:
        return {"products": {"Product_A": {"positive_topics": {}, "negative_topics": {}}}}
    combined_text = "\n".join(map(str, reviews))
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_text},
        ],
        stream=False
    )
    content = response.choices[0].message.content
    content = clean_api_response(content)
    result = json.loads(content)
    return result

def generate_wordcloud_from_json(json_data):
    """
    Generate a word cloud from the API JSON output by summing frequencies
    across positive and negative topics.
    """
    frequencies = {}
    products = json_data.get("products", {})
    for product, product_data in products.items():
        for sentiment in ["positive_topics", "negative_topics"]:
            topics = product_data.get(sentiment, {})
            for topic, freq in topics.items():
                frequencies[topic] = frequencies.get(topic, 0) + freq
    if not frequencies:
        return None
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(frequencies)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.close(fig)
    return fig

#############################################
# INTEGRATED PROCESSING FUNCTIONS FOR GRADIO
#############################################
def process_single_text(text):
    """
    For single text input:
      - Perform sentiment analysis (predicted sentiment and confidence pie chart)
      - Call the extraction API to get JSON topics and generate a word cloud.
    Returns:
      - Predicted sentiment (text)
      - Confidence pie chart (plot)
      - API JSON response (text)
      - Word cloud (plot)
    """
    # Sentiment analysis
    sentiment, conf_fig = predict_sentiment(text)

    # API extraction and word cloud
    try:
        api_result = get_keywords_from_api(text)
        json_output = json.dumps(api_result, indent=2)
    except Exception as e:
        json_output = json.dumps({"error": str(e)}, indent=2)
        api_result = {}
    wordcloud_fig = generate_wordcloud_from_json(api_result) if "error" not in api_result else None
    return sentiment, conf_fig, json_output, wordcloud_fig

def process_csv_file(file_obj, column_name):
    """
    For CSV input:
      - Generate a sentiment distribution pie chart using the ensemble model.
      - Call the extraction API on the CSV reviews and generate a word cloud.
    Returns:
      - Sentiment distribution pie chart (plot)
      - API JSON response (text)
      - Word cloud (plot)
    """
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        return {"error": f"Error reading CSV file: {e}"}, None, None
    if len(df) > 100:
        df = df.head(100)
    text_col = column_name if column_name and column_name in df.columns else df.columns[0]

    # Generate sentiment distribution pie chart using CSV file text
    sent_fig = predict_sentiment_csv(file_obj, column_name)

    # API extraction and word cloud from CSV reviews
    try:
        api_result = get_keywords_from_api(df[text_col])
        json_output = json.dumps(api_result, indent=2)
    except Exception as e:
        json_output = json.dumps({"error": str(e)}, indent=2)
        api_result = {}
    wordcloud_fig = generate_wordcloud_from_json(api_result) if "error" not in api_result else None
    return sent_fig, json_output, wordcloud_fig

#############################################
# GRADIO INTERFACE SETUP (TABBED)
#############################################
# Single Text Analysis tab: returns sentiment, confidence pie chart, API JSON, and word cloud.
single_text_interface = gr.Interface(
    fn=process_single_text,
    inputs=gr.Textbox(
        label="Enter Text Reviews",
        placeholder="Type your text reviews here...",
        lines=5
    ),
    outputs=[
        gr.Textbox(label="Predicted Sentiment"),
        gr.Plot(label="Confidence Pie Chart"),
        gr.JSON(label="API JSON Response"),
        gr.Plot(label="Word Cloud")
    ],
    title="Single Text Analysis",
    description="Enter text reviews to get sentiment classification, topic extraction, and a word cloud."
)

# CSV Analysis tab: returns sentiment distribution pie chart, API JSON, and word cloud.
csv_interface = gr.Interface(
    fn=process_csv_file,
    inputs=[
        gr.File(label="Upload CSV File", file_types=['.csv'], type="filepath"),
        gr.Textbox(label="Column Name", placeholder="Enter column name containing text (or leave blank for default)")
    ],
    outputs=[
        gr.Plot(label="Sentiment Distribution Pie Chart"),
        gr.JSON(label="API JSON Response"),
        gr.Plot(label="Word Cloud")
    ],
    title="CSV Analysis",
    description="Upload a CSV file containing reviews to get sentiment distribution, topic extraction, and a word cloud."
)

tabbed_interface = gr.TabbedInterface(
    [single_text_interface, csv_interface],
    ["Single Text Analysis", "CSV Analysis"]
)

tabbed_interface.launch()

