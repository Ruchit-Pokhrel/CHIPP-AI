import gradio as gr
import pandas as pd
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from transformers import pipeline
import torch
import openai
from dotenv import load_dotenv
import os
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenize
from typing import Union

 #Load environment variables
load_dotenv(dotenv_path='/content/drive/MyDrive/.env')
API_KEY = os.getenv('API_KEY')

# Set device for model inference
device = 0 if torch.cuda.is_available() else -1

model_names = ["CHIPP-AI/model1","CHIPP-AI/model2","CHIPP-AI/model3"]

system_prompt = """

Using the provided reviews, extract key aspects, assign polarity, and categorize them into topics. Replace the original words with their respective topics and generate a JSON output that includes the frequency of each topic. The JSON will be used to create a word cloud, where the size of each word corresponds to its frequency. Provide only json output.
Example output:
{
  "products": {
    "Product_A": {
      "positive_topics": {
        "Performance": 8,
        "Reliability": 7,
        "Delivery Times":3
      },
      "negative_topics": {
        "Build Quality": 2,
        "Delivery Times":1
      }
    },
    "Product_B": {
      "positive_topics": {
        "Speed": 6,
        "Durability": 5
      },
      "negative_topics": {
        "Write Speed": 3
      }
    }
  }
}


"""

def soft_voting_ensemble(text, models, tokenizer):
    """
    Perform soft voting ensemble on the given text.
    
    Args:
        text (str): Input text for classification.
        models (list): List of transformer models.
        tokenizer (AutoTokenizer): Tokenizer for text preprocessing.

    Returns:
        dict: Averaged softmax probabilities and the final predicted label.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Collect probabilities from all models
    all_probs = []
    
    for model in models:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            all_probs.append(probs)
    
    # Stack and average probabilities
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    final_label = torch.argmax(avg_probs, dim=-1).item()

    return {"avg_probs": avg_probs.tolist()[0], "predicted_label": final_label}

 print("how did chippy become so dumb")   
