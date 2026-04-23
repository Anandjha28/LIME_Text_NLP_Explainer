import json
import random
from typing import Dict, List, Any
import streamlit as st

def load_sample_texts(filepath: str = "data/sample_texts.json") -> Dict:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"sentiment_examples": [
            {"text": "This movie was fantastic!", "label": "positive"},
            {"text": "Terrible product.", "label": "negative"}
        ]}

def get_random_example(category: str = "sentiment") -> Dict:
    samples = load_sample_texts()
    examples = samples.get("sentiment_examples", [])
    if examples:
        return random.choice(examples)
    return {"text": "Sample text", "label": "neutral"}

def format_confidence(confidence: float) -> str:
    return f"{confidence * 100:.1f}%"

def truncate_text(text: str, max_length: int = 200) -> str:
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text