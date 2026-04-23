import streamlit as st
import os
import pandas as pd
import numpy as np

# ================ PAGE CONFIG MUST BE FIRST ================
st.set_page_config(
    page_title="LIME NLP Explainer",
    page_icon="🍋",
    layout="wide"
)

# ================ CUSTOM CSS ================
st.markdown("""
<style>
    .main-title {
        color: #FF4B4B;
        padding-bottom: 1rem;
    }
    .highlight-box {
        background-color: #1a1a1a;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .positive-word {
        background-color: #d4edda !important;
        color: #155724 !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        border: 2px solid #28a745 !important;
        font-weight: bold !important;
        margin: 2px !important;
        display: inline-block !important;
    }
    .negative-word {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        border: 2px solid #dc3545 !important;
        font-weight: bold !important;
        margin: 2px !important;
        display: inline-block !important;
    }
    .example-button {
        margin: 5px 0 !important;
    }
    .fast-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ================ SESSION STATE INIT ================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_exp' not in st.session_state:
    st.session_state.current_exp = None
if 'example_text' not in st.session_state:
    st.session_state.example_text = ""
if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False

# ================ CACHED FUNCTIONS ================

@st.cache_resource
def load_sentiment_model():
    """Load once, use everywhere - FAST!"""
    from transformers import pipeline
    # Using DistilBERT for speed (6x faster than BERT)
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )

@st.cache_data(ttl=3600, show_spinner="Generating explanation...")
def get_lime_explanation(_model, text, num_features=10, num_samples=2000):
    """Cached LIME explanation"""
    import lime.lime_text
    
    explainer = lime.lime_text.LimeTextExplainer(
        class_names=['NEGATIVE', 'POSITIVE'],
        bow=False,  # Faster without bag-of-words
        split_expression=lambda x: x.split()
    )
    
    def predict_proba(texts):
        """Batch prediction for speed"""
        batch_results = _model(texts, truncation=True)
        predictions = []
        
        for res in batch_results:
            label = res['label'].upper()
            score = res['score']
            
            if "POS" in label:
                pos_score = score
                neg_score = 1 - score
            else:
                neg_score = score
                pos_score = 1 - score
            
            predictions.append([neg_score, pos_score])
        
        return np.array(predictions)
    
    # Reduced samples for speed (2000 instead of 5000)
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,
        num_samples=num_samples
    )
    
    return exp

@st.cache_data(ttl=3600)
def precompute_examples(_model, examples):
    """Pre-compute example predictions"""
    results = {}
    for example in examples:
        result = _model(example, truncation=True)[0]
        results[example] = {
            'label': result['label'],
            'score': result['score']
        }
    return results

# ================ HEADER ================
st.markdown('<h1 class="main-title">🍋 LIME Text NLP Explainer <span class="fast-badge"></span></h1>', unsafe_allow_html=True)
st.markdown("""
<div class="highlight-box">
<h4>⚡ Understand AI Predictions with LIME</h4>
<p>Visualize which words influence sentiment analysis predictions. <strong></strong></p>
</div>
""", unsafe_allow_html=True)

# ================ SIDEBAR ================
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_name = st.selectbox(
        "Choose Model",
        ["DistilBERT (Fast)", "BERT (Accurate)"],
        index=0,
        help="DistilBERT is 6x faster with minimal accuracy loss"
    )
    
    num_features = st.slider(
        "Features to show",
        min_value=5,
        max_value=15,  # Reduced for speed
        value=8,
        help="Fewer features = faster processing"
    )
    
    st.markdown("---")
    st.header("📚 Examples")
    st.caption("Click to load instantly (pre-computed)")
    
    examples = [
        "I love this movie! It was absolutely fantastic.",
        "Terrible service, would not recommend.",
        "The product is okay, nothing special.",
        "Excellent quality, highly recommended!",
        "Very disappointed with the purchase."
    ]
    
    # Load model once for examples
    example_model = load_sentiment_model()
    example_results = precompute_examples(example_model, examples)
    
    for i, example in enumerate(examples):
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True, type="secondary"):
                st.session_state.example_text = example
                st.session_state.current_exp = None
                st.rerun()
        with col2:
            # Show quick prediction badge
            pred = example_results[example]
            label = "😊" if "POS" in pred['label'].upper() else "😞"
            st.caption(label)

# ================ MAIN CONTENT ================
st.subheader("📝 Enter Text to Analyze")

# Text input with auto-clear
text = st.text_area(
    "",
    value=st.session_state.get('example_text', "I love this movie! It was absolutely fantastic."),
    height=100,
    placeholder="Type or select an example above...",
    help="Press Ctrl+Enter to analyze quickly"
)

# Analyze button with better feedback
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    analyze_clicked = st.button(
        "🔍 Analyze with LIME",
        type="primary",
        use_container_width=True,
        disabled=(not text.strip() or st.session_state.get('is_loading', False))
    )
with col2:
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.example_text = ""
        st.session_state.current_exp = None
        st.rerun()

# ================ PROCESS ANALYSIS ================
if analyze_clicked and text.strip():
    st.session_state.is_loading = True
    
    # Progress container
    progress_container = st.empty()
    with progress_container.container():
        st.markdown("### ⏳ Processing...")
        
        # Progress steps
        progress_steps = st.progress(0, text="Starting analysis...")
        
        try:
            # Step 1: Load model (cached)
            progress_steps.progress(20, text="Loading model (cached)...")
            model = load_sentiment_model()
            
            # Step 2: Get quick prediction
            progress_steps.progress(40, text="Getting sentiment...")
            result = model(text, truncation=True)[0]
            initial_label = "POSITIVE" if "POS" in result['label'].upper() else "NEGATIVE"
            initial_conf = result['score']
            
            # Show quick result
            st.success(f"✅ **Quick Prediction:** {initial_label} ({initial_conf*100:.1f}%)")
            
            # Step 3: LIME explanation (cached)
            progress_steps.progress(60, text="Generating explanation...")
            
            # Use cached LIME function
            exp = get_lime_explanation(
                model, 
                text, 
                num_features=num_features,
                num_samples=2000  # Reduced for speed
            )
            
            # Step 4: Process results
            progress_steps.progress(80, text="Processing visualization...")
            
            # Extract features
            exp_map = exp.as_map()
            features = []
            
            if 1 in exp_map:  # Positive class
                for idx, weight in exp_map[1]:
                    if idx < len(exp.domain_mapper.indexed_string):
                        word = exp.domain_mapper.indexed_string[idx]
                        # Filter out very small weights
                        if abs(weight) > 0.001:
                            features.append({
                                'word': word,
                                'weight': float(weight),
                                'abs_weight': abs(weight)
                            })
            
            # Sort and limit
            features.sort(key=lambda x: x['abs_weight'], reverse=True)
            top_features = features[:num_features]
            
            # Get LIME prediction
            label_idx = exp.local_pred.argmax()
            lime_label = "POSITIVE" if label_idx == 1 else "NEGATIVE"
            lime_conf = exp.local_pred[label_idx]
            
            # Store in session
            explanation = {
                'text': text,
                'prediction': lime_label,
                'confidence': lime_conf,
                'features': top_features,
                'pos_contrib': sum(f['weight'] for f in top_features if f['weight'] > 0),
                'neg_contrib': sum(f['weight'] for f in top_features if f['weight'] < 0),
                'timestamp': pd.Timestamp.now().strftime("%H:%M:%S")
            }
            
            st.session_state.current_exp = explanation
            if explanation not in st.session_state.history:
                st.session_state.history.append(explanation)
            
            # Clear progress
            progress_steps.progress(100, text="Complete!")
            
            # Remove progress container
            progress_container.empty()
            
            # ================ DISPLAY RESULTS ================
            st.markdown("---")
            st.subheader("📊 LIME Explanation Results")
            
            # Quick stats row
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Prediction", lime_label)
            with col_stat2:
                st.metric("Confidence", f"{lime_conf*100:.1f}%")
            with col_stat3:
                pos_words = len([f for f in top_features if f['weight'] > 0])
                st.metric("Positive Words", pos_words)
            with col_stat4:
                neg_words = len([f for f in top_features if f['weight'] < 0])
                st.metric("Negative Words", neg_words)
            
            # Word highlights
            st.markdown("#### 🔍 Word Impact Analysis")
            
            if top_features:
                # Positive words
                positive_words = [f for f in top_features if f['weight'] > 0]
                if positive_words:
                    st.markdown("**✅ Supporting words:**")
                    pos_html = ""
                    for feature in positive_words:
                        word = feature['word']
                        weight = feature['weight']
                        pos_html += f'<span class="positive-word">{word} (+{weight:.3f})</span> '
                    st.markdown(pos_html, unsafe_allow_html=True)
                
                # Negative words
                negative_words = [f for f in top_features if f['weight'] < 0]
                if negative_words:
                    st.markdown("**❌ Opposing words:**")
                    neg_html = ""
                    for feature in negative_words:
                        word = feature['word']
                        weight = feature['weight']
                        neg_html += f'<span class="negative-word">{word} ({weight:.3f})</span> '
                    st.markdown(neg_html, unsafe_allow_html=True)
            
            # Original text box
            st.markdown("---")
            with st.expander("📝 View Original Text", expanded=True):
                st.write(text)
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Feature Contributions")
            
            if top_features:
                # Create simple bar chart
                import plotly.graph_objects as go
                
                df = pd.DataFrame(top_features)
                df = df.sort_values('weight', ascending=True)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    y=df['word'],
                    x=df['weight'],
                    orientation='h',
                    marker_color=['#28a745' if w > 0 else '#dc3545' for w in df['weight']],
                    text=[f"{w:+.3f}" for w in df['weight']],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title=f"Top {len(df)} Influential Words",
                    xaxis_title="Contribution Score",
                    yaxis_title="Word",
                    showlegend=False,
                    height=max(300, len(df) * 30),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                with st.expander("📋 Detailed Data Table"):
                    display_df = df.copy()
                    display_df = display_df[['word', 'weight']]
                    display_df.columns = ['Word', 'Contribution']
                    display_df['Effect'] = display_df['Contribution'].apply(
                        lambda x: 'Positive' if x > 0 else 'Negative'
                    )
                    st.dataframe(display_df, use_container_width=True)
            
            st.balloons()
            st.success("✨ Analysis complete in seconds!")
            
        except Exception as e:
            progress_container.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Debug Details"):
                st.code(traceback.format_exc())
    
    st.session_state.is_loading = False

elif analyze_clicked and not text.strip():
    st.warning("⚠️ Please enter some text to analyze.")

# ================ HISTORY SIDEBAR ================
with st.sidebar:
    st.markdown("---")
    st.subheader("📋 Recent Analyses")
    
    if st.session_state.history:
        # Show last 3 analyses
        for i, item in enumerate(reversed(st.session_state.history[-3:])):
            idx = len(st.session_state.history) - i
            with st.expander(f"Analysis #{idx} - {item['timestamp']}", expanded=False):
                st.caption(f"**Text:** {item['text'][:40]}...")
                st.caption(f"**Prediction:** {item['prediction']}")
                st.caption(f"**Confidence:** {item['confidence']*100:.1f}%")
                
                if st.button(f"📝 Load this text", key=f"load_{idx}", use_container_width=True):
                    st.session_state.example_text = item['text']
                    st.rerun()
                    
                if st.button(f"🗑️ Remove", key=f"remove_{idx}", use_container_width=True, type="secondary"):
                    st.session_state.history.pop(len(st.session_state.history) - i - 1)
                    st.rerun()
    else:
        st.info("No analyses yet. Try one!")
    
    # Clear all button
    if st.session_state.history:
        if st.button("🗑️ Clear All History", use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.rerun()

# ================ PERFORMANCE INFO ================
with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚡ Performance Tips")
    st.markdown("""
    1. **Examples load instantly** (pre-computed)
    2. **Same text?** Uses cache (no recomputation)
    3. **Short text** = faster analysis
    4. **Fewer features** = quicker results
    """)

# ================ FOOTER ================
st.markdown("---")
st.caption("🚀 Built with Streamlit, Transformers, and LIME | **Optimized for Speed** | Interactive NLP Dashboard")