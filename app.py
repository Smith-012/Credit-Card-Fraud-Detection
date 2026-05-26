"""
Streamlit Interactive App - Credit Card Fraud Detection
Shows model predictions with SHAP explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3em; color: #9467bd; font-weight: bold; text-align: center; margin-bottom: 0.5em; }
    .metric-card { background-color: #f0f2f6; padding: 1.5em; border-radius: 0.5em; border-left: 4px solid #9467bd; }
    .prediction-box { background-color: #f3e6ff; padding: 2em; border-radius: 0.5em; text-align: center; }
    .fraud { color: #d62728; font-weight: bold; font-size: 2em; }
    .legit { color: #2ca02c; font-weight: bold; font-size: 2em; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚨 Credit Card Fraud Detector</div>', unsafe_allow_html=True)

# Load model and preprocessor
@st.cache_resource
def load_model_and_data():
    artifact = joblib.load("artifacts/model.joblib")
    if isinstance(artifact, dict):
        model = artifact
    else:
        model = {'model': artifact}
    df_train = pd.read_csv("data/creditcard.csv")
    return model, df_train

try:
    model_artifact, df_train = load_model_and_data()
except FileNotFoundError:
    st.error("⚠️ Model not found! Please run `python main.py --data data/creditcard.csv --out artifacts` first.")
    st.stop()

# Sidebar - Input features
st.sidebar.header("💳 Transaction Details")

col1, col2 = st.sidebar.columns(2)
with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=5000.0, value=100.0)
    time_of_day = st.selectbox("Time of Day", ["00:00-06:00 (Night)", "06:00-12:00 (Morning)", "12:00-18:00 (Afternoon)", "18:00-00:00 (Evening)"])
    transaction_type = st.selectbox("Transaction Type", ["Purchase", "Cash Withdrawal", "Transfer", "Online"])

with col2:
    merchant_risk = st.slider("Merchant Risk Level", min_value=0, max_value=10, value=5)
    frequency = st.slider("Transaction Frequency (per day)", min_value=0, max_value=50, value=5)

# Create synthetic feature vector (simplified for demo)
# In practice, you'd encode these properly
input_data = pd.DataFrame({
    'Amount': [amount],
    'V1': [merchant_risk / 10],
    'V2': [frequency / 50],
    'V3': [time_of_day_encode(time_of_day)],
    'V4': [0.5],  # Placeholder features
    'V5': [0.3],
    'V6': [0.1],
    'V7': [0.2],
    'V8': [0.4],
    'V9': [0.6],
})

def time_of_day_encode(time_str):
    """Encode time of day as numerical value"""
    if "Night" in time_str:
        return -0.5
    elif "Morning" in time_str:
        return 0.0
    elif "Afternoon" in time_str:
        return 0.3
    else:
        return 0.1

# Main prediction
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Fraud Status")
    
    try:
        # Extract model components
        if isinstance(model_artifact, dict):
            preprocessor = model_artifact.get('preprocessor')
            classifier = model_artifact.get('model')
        else:
            if hasattr(model_artifact, 'named_steps'):
                preprocessor = model_artifact.named_steps.get('preprocessor')
                classifier = model_artifact.named_steps.get('clf')
            else:
                preprocessor = None
                classifier = model_artifact
        
        # Transform input
        if preprocessor:
            try:
                input_transformed = preprocessor.transform(input_data)
            except:
                input_transformed = input_data
        else:
            input_transformed = input_data
        
        # Get prediction and probability
        prediction = classifier.predict(input_transformed)[0]
        probability = classifier.predict_proba(input_transformed)[0]
        fraud_probability = probability[1] if len(probability) > 1 else probability[0]
        
        # Display prediction
        if prediction == 1:
            st.markdown('<div class="prediction-box"><div class="fraud">⚠️ FRAUDULENT</div></div>', unsafe_allow_html=True)
            prob_text = f"Fraud Probability: **{fraud_probability:.1%}**"
            st.error(prob_text)
        else:
            st.markdown('<div class="prediction-box"><div class="legit">✓ LEGITIMATE</div></div>', unsafe_allow_html=True)
            prob_text = f"Fraud Probability: **{fraud_probability:.1%}**"
            st.success(prob_text)
        
        # Risk assessment
        if fraud_probability > 0.8:
            st.error("🚨 **HIGH RISK** - Block transaction immediately")
        elif fraud_probability > 0.5:
            st.warning("⚠️ **MEDIUM RISK** - Verify with customer")
        else:
            st.success("✓ **LOW RISK** - Safe to approve")
        
        # Model metrics
        st.markdown("### Model Performance")
        metrics_file = Path("artifacts/metrics.txt")
        if metrics_file.exists():
            metrics_text = metrics_file.read_text()
            for line in metrics_text.split('\n')[:7]:
                if line.strip():
                    st.text(line)
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

with col2:
    st.subheader("🔍 SHAP Explainability")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(classifier) if hasattr(classifier, 'tree_') else shap.KernelExplainer(classifier.predict_proba, input_transformed[:1])
        shap_values = explainer.shap_values(input_transformed)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
        
        # Display feature importance
        fig, ax = plt.subplots(figsize=(10, 4))
        feature_names = [f'V{i}' for i in range(1, 10)] + ['Amount']
        
        # Show top features
        importance = np.abs(shap_vals[0])
        importance_indices = np.argsort(importance)[-5:]
        top_features = [feature_names[i] if i < len(feature_names) else f'Feature{i}' for i in importance_indices]
        top_values = shap_vals[0][importance_indices]
        
        colors = ['red' if v > 0 else 'green' for v in top_values]
        ax.barh(range(len(top_features)), top_values, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('SHAP Value (Impact on Fraud Score)')
        ax.set_title('Top Features Influencing Fraud Detection')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
    
    except Exception as e:
        st.warning(f"SHAP visualization: {str(e)}")
        st.info("📊 Feature importance loading...")

# Dataset Info
st.markdown("---")
st.subheader("📈 Dataset Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", len(df_train))
    fraud_count = (df_train['Class'] == 1).sum() if 'Class' in df_train.columns else 0
    st.metric("Fraud Cases", f"{fraud_count} ({fraud_count/len(df_train):.2%})")

with col2:
    st.metric("Avg Transaction", f"${df_train['Amount'].mean():.2f}")
    st.metric("Max Transaction", f"${df_train['Amount'].max():.2f}")

with col3:
    st.metric("Min Transaction", f"${df_train['Amount'].min():.2f}")
    st.metric("Std Deviation", f"${df_train['Amount'].std():.2f}")

st.markdown("""
---
**📚 About This App:**
- Built with Streamlit for interactive predictions
- Uses SHAP for model explainability
- Detects credit card fraud with high precision
- Model: Random Forest Classifier with SMOTE for class imbalance
- Alerts: Color-coded risk levels for quick decision-making
""")
