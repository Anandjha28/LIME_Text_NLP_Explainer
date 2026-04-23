import lime
import lime.lime_text
import numpy as np
from typing import Dict, List, Any, Tuple
import streamlit as st

class LimeExplainer:
    """LIME explanation generator for text models."""
    
    def __init__(self, class_names: List[str] = None):
        if class_names is None:
            class_names = ['negative', 'positive']
        self.class_names = class_names
        self.explainer = lime.lime_text.LimeTextExplainer(
            class_names=class_names,
            split_expression=lambda x: x.split(),
            bow=False
        )
    
    def create_predict_function(self, model, model_type: str = 'sentiment') -> callable:
        """Create a prediction function for LIME."""
        def predict_proba(texts: List[str]) -> np.ndarray:
            predictions = []
            for text in texts:
                result = model(text)[0]
                
                if model_type == 'sentiment':
                    # Handle sentiment models
                    label = result['label'].lower()
                    score = result['score']
                    
                    if 'pos' in label or 'positive' in label:
                        pos_score = score
                        neg_score = 1 - score
                    elif 'neg' in label or 'negative' in label:
                        neg_score = score
                        pos_score = 1 - score
                    else:
                        # For neutral or other labels
                        pos_score = score if 'pos' in label else 0.5
                        neg_score = 1 - pos_score
                    
                    predictions.append([neg_score, pos_score])
                
                elif model_type == 'emotion':
                    # Handle multi-class emotion models
                    # This is simplified - you'd need to adapt for actual multi-class
                    predictions.append([0.3, 0.7])  # Placeholder
                
                else:
                    # Default binary classification
                    score = result['score']
                    label = result['label']
                    predictions.append([1 - score, score])
            
            return np.array(predictions)
        
        return predict_proba
    
    def explain(self, 
                text: str, 
                model, 
                model_type: str = 'sentiment',
                num_features: int = 10,
                num_samples: int = 5000) -> Dict:
        """Generate LIME explanation for a text."""
        
        # Create prediction function
        predict_fn = self.create_predict_function(model, model_type)
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Parse explanation into structured format
        exp_map = explanation.as_map()
        
        # Get label with highest probability
        label_idx = explanation.local_pred.argmax()
        predicted_label = self.class_names[label_idx] if label_idx < len(self.class_names) else str(label_idx)
        
        # Extract feature contributions
        features = []
        if 1 in exp_map:  # Positive class features (index 1)
            for idx, weight in exp_map[1]:
                if idx < len(explanation.domain_mapper.indexed_string):
                    word = explanation.domain_mapper.indexed_string[idx]
                    features.append({
                        'word': word,
                        'weight': float(weight),
                        'contribution': 'positive' if weight > 0 else 'negative',
                        'abs_weight': abs(weight)
                    })
        
        # Sort by absolute weight
        features.sort(key=lambda x: x['abs_weight'], reverse=True)
        
        # Get top features
        top_features = features[:num_features]
        
        # Calculate statistics
        total_pos = sum(f['weight'] for f in top_features if f['weight'] > 0)
        total_neg = sum(f['weight'] for f in top_features if f['weight'] < 0)
        
        return {
            'predicted_label': predicted_label,
            'prediction_confidence': float(explanation.local_pred[label_idx]),
            'features': top_features,
            'total_positive_contrib': total_pos,
            'total_negative_contrib': total_neg,
            'explanation_object': explanation,
            'intercept': float(explanation.intercept[1]) if hasattr(explanation, 'intercept') else 0.0
        }
    
    def get_word_highlights(self, text: str, features: List[Dict]) -> str:
        """Generate HTML with highlighted words based on contributions."""
        # Sort features by word length (longer words first) to avoid partial highlighting
        sorted_features = sorted(features, key=lambda x: len(x['word']), reverse=True)
        
        highlighted_text = text
        
        for feature in sorted_features:
            word = feature['word']
            weight = feature['weight']
            
            # Determine color based on weight
            if weight > 0:
                # Green for positive contributions
                intensity = min(0.5 + abs(weight) * 2, 1.0)
                color = f'rgba(0, {int(200 * intensity)}, 0, {0.3 + intensity * 0.7})'
            else:
                # Red for negative contributions
                intensity = min(0.5 + abs(weight) * 2, 1.0)
                color = f'rgba({int(200 * intensity)}, 0, 0, {0.3 + intensity * 0.7})'
            
            # Create highlighted span
            highlight = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 0 1px;">{word}</span>'
            
            # Replace word in text (case insensitive)
            import re
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted_text = pattern.sub(highlight, highlighted_text)
        
        return f'<div style="line-height: 1.8; font-size: 16px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">{highlighted_text}</div>'