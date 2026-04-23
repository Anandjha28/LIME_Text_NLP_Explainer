import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self):
        self.color_positive = '#2ecc71'
        self.color_negative = '#e74c3c'
        self.color_neutral = '#3498db'
    
    def create_feature_plot(self, features, title="Feature Contributions"):
        if not features:
            return None
        
        df = pd.DataFrame(features)
        df = df.sort_values('weight', ascending=True)
        
        colors = [self.color_positive if w > 0 else self.color_negative for w in df['weight']]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df['word'], x=df['weight'], orientation='h',
            marker_color=colors,
            text=[f"{w:+.3f}" for w in df['weight']], textposition='auto'
        ))
        
        fig.update_layout(
            title=title, xaxis_title="Contribution", yaxis_title="Word",
            showlegend=False, height=400, yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def create_confidence_gauge(self, confidence: float, label: str):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Confidence: {label.upper()}"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': self.color_positive if confidence > 0.5 else self.color_negative}}
        ))
        
        fig.update_layout(height=250)
        return fig
    
    def create_wordcloud(self, features):
        if not features:
            return None
        
        word_freq = {f['word']: f.get('abs_weight', 1) * 100 for f in features}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Feature Importance Word Cloud')
        
        return fig
    
    def create_summary_stats(self, explanation):
        # Extract values safely
        features = explanation.get('features', [])
        pos_contrib = sum(f.get('weight', 0) for f in features if f.get('weight', 0) > 0)
        neg_contrib = sum(f.get('weight', 0) for f in features if f.get('weight', 0) < 0)
        
        stats_data = [
            ['Positive', pos_contrib],
            ['Negative', neg_contrib],
            ['Features', len(features)]
        ]
        
        df = pd.DataFrame(stats_data, columns=['Metric', 'Value'])
        
        colors = [self.color_positive, self.color_negative, self.color_neutral]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['Metric'], y=df['Value'],
            marker_color=colors,
            text=[f"{v:.3f}" if isinstance(v, float) else str(v) for v in df['Value']],
            textposition='auto'
        ))
        
        fig.update_layout(title="Summary Statistics", showlegend=False, height=300)
        return fig