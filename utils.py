import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv
from wordcloud import WordCloud

# Load environment variables from a local .env file (if present) or Hugging Face Spaces secrets
load_dotenv()

API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables")

# Use an environment variable for the API base URL; defaults to the Deepseek API URL.
DEEPSEEK_API_BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")
client = OpenAI(api_key=API_KEY, base_url=DEEPSEEK_API_BASE_URL)

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
    fig = px.pie(values=sizes, names=labels, title="Confidence Scores (%)")
    return sentiment, fig

def predict_sentiment_csv(file_obj, column_name):
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        return {"error": f"Error reading CSV file: {e}"}
    if len(df) > 100:
        df = df.head(100)
    text_col = column_name if column_name and column_name in df.columns else df.columns[0]
    counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
    for text in df[text_col]:
        if not isinstance(text, str):
            text = str(text)
        if not text.strip():
            continue
        sentiment, _ = model.predict(text)
        counts[sentiment] += 1
    labels = list(counts.keys())
    sizes = list(counts.values())
    fig = px.pie(values=sizes, names=labels, title="Sentiment Distribution (First 100 Rows)")
    return fig

#############################################
# API EXTRACTION & WORD CLOUD CODE
#############################################
def process_input(input_data):
    """Convert input to a list of texts."""
    if isinstance(input_data, pd.DataFrame):
        text_col = "text" if "text" in input_data.columns else input_data.columns[0]
        return input_data[text_col].fillna("").tolist()
    elif isinstance(input_data, pd.Series):
        return input_data.fillna("").tolist()
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
As an AI text analyzer, your task is to extract key topics and sentiments from the provided reviews. Do not categorise products.
For example, IPhone, Samsung, Huawei should be their own products instead of categorising them as mobile phones.
Generate a JSON output with the following structure:
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

def format_topics_for_product(api_json, product):
    """Return a formatted string of topics for a single product."""
    lines = []
    products = api_json.get("products", {})
    if product in products:
        data = products[product]
        lines.append(f"Product: {product}")
        lines.append("  Positive Topics:")
        pos = data.get("positive_topics", {})
        if pos:
            for topic, freq in pos.items():
                lines.append(f"    - {topic}: {freq}")
        else:
            lines.append("    None")
        lines.append("  Negative Topics:")
        neg = data.get("negative_topics", {})
        if neg:
            for topic, freq in neg.items():
                lines.append(f"    - {topic}: {freq}")
        else:
            lines.append("    None")
    else:
        lines.append("No topics found for selected product.")
    return "\n".join(lines)

def generate_wordclouds_by_product_plotly(json_data):
    """Generate wordclouds for CSV analysis."""
    product_wordclouds = {}
    products = json_data.get("products", {})
    for prod, data in products.items():
        pos = data.get("positive_topics", {})
        neg = data.get("negative_topics", {})
        # Positive wordcloud
        if pos:
            wc_pos = WordCloud(width=400, height=300, background_color="white").generate_from_frequencies(pos)
            pos_img = wc_pos.to_array()
            fig_pos = px.imshow(pos_img, template="plotly_white")
            fig_pos.update_layout(coloraxis_showscale=False, title_text=f"{prod} - Positive")
        else:
            blank = np.zeros((300, 400, 3), dtype=np.uint8)
            fig_pos = px.imshow(blank, template="plotly_white")
            fig_pos.add_annotation(text="No positive data", x=0.5, y=0.5, xref="paper", yref="paper",
                                   showarrow=False, font=dict(size=20))
            fig_pos.update_layout(title_text=f"{prod} - Positive", coloraxis_showscale=False)
        # Negative wordcloud
        if neg:
            wc_neg = WordCloud(width=400, height=300, background_color="white").generate_from_frequencies(neg)
            neg_img = wc_neg.to_array()
            fig_neg = px.imshow(neg_img, template="plotly_white")
            fig_neg.update_layout(coloraxis_showscale=False, title_text=f"{prod} - Negative")
        else:
            blank = np.zeros((300, 400, 3), dtype=np.uint8)
            fig_neg = px.imshow(blank, template="plotly_white")
            fig_neg.add_annotation(text="No negative data", x=0.5, y=0.5, xref="paper", yref="paper",
                                   showarrow=False, font=dict(size=20))
            fig_neg.update_layout(title_text=f"{prod} - Negative", coloraxis_showscale=False)
        product_wordclouds[prod] = (fig_pos, fig_neg)
    return product_wordclouds

#############################################
# PROCESSING FUNCTIONS FOR SINGLE TEXT & CSV WITH PRODUCTS
#############################################
def process_single_text_with_products(text):
    """Process single text input without wordcloud generation."""
    sentiment, conf_fig = predict_sentiment(text)
    try:
        api_result = get_keywords_from_api(text)
    except Exception as e:
        api_result = {}
    product_names = list(api_result.get("products", {}).keys())
    dropdown_update = {"choices": product_names, "value": product_names[0] if product_names else None}
    state = {"api_json": api_result}
    return sentiment, conf_fig, dropdown_update, state

def process_csv_file_with_products(file_obj, column_name):
    """Process CSV file with wordcloud generation."""
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        return {"error": f"Error reading CSV file: {e}"}, {"choices": []}, None
    if len(df) > 100:
        df = df.head(100)
    text_col = column_name if column_name and column_name in df.columns else df.columns[0]
    sent_fig = predict_sentiment_csv(file_obj, column_name)
    try:
        api_result = get_keywords_from_api(df[text_col])
    except Exception as e:
        api_result = {}
    wc_dict = generate_wordclouds_by_product_plotly(api_result)
    product_names = list(wc_dict.keys()) if wc_dict else []
    dropdown_update = {"choices": product_names, "value": product_names[0] if product_names else None}
    state = {"api_json": api_result, "wc_dict": wc_dict}
    return sent_fig, dropdown_update, state

def display_product_info(selected_product, state):
    """Display product information based on context."""
    if not state or not selected_product:
        return "No data available", None, None

    topics_text = format_topics_for_product(state.get("api_json", {}), selected_product)

    if "wc_dict" in state and selected_product in state["wc_dict"]:
        pos_wc, neg_wc = state["wc_dict"][selected_product]
        return topics_text, pos_wc, neg_wc
    else:
        return topics_text, None, None
