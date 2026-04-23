import streamlit as st
from transformers import pipeline
import joblib
import os
from typing import Dict, Any, Optional, List

class ModelLoader:
    def __init__(self):
        self.models = {}
        self.available_models = {
            'sentiment': {
                'bert': 'distilbert-base-uncased-finetuned-sst-2-english',
                'roberta': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english'
            }
        }
    
    @st.cache_resource(show_spinner=False)
    def load_transformers_model(_self, model_name: str):
        try:
            return pipeline("sentiment-analysis", model=model_name)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    def get_model(self, model_type: str = 'sentiment', model_name: str = 'bert'):
        cache_key = f"{model_type}_{model_name}"
        if cache_key not in self.models:
            if model_type in self.available_models and model_name in self.available_models[model_type]:
                model_id = self.available_models[model_type][model_name]
                model = self.load_transformers_model(model_id)
                if model:
                    self.models[cache_key] = model
        return self.models.get(cache_key)