# Credit Card Fraud Detection

## Portfolio Summary
This version turns the fraud detection task into a stronger portfolio project by addressing class imbalance, comparing models, and storing the full training result as artifacts.

## What Makes It Portfolio-Ready

- Class imbalance handling with SMOTE
- Model comparison using cross-validation on F1
- Fraud-centric evaluation metrics
- Saved preprocessing and model artifact

## 🚨 What This Project Does

This project detects fraudulent credit card transactions using machine learning. Given transaction features (amount, merchant category, time, location patterns, etc.), the model classifies whether a transaction is legitimate or fraudulent. It demonstrates:

- **Class imbalance handling**: Using SMOTE to balance minority fraud cases
- **Domain-specific metrics**: Using Precision, Recall, and F1 score instead of accuracy
- **Risk-aware prediction**: Understanding false negatives (missed fraud) cost more than false positives
- **Real-time preprocessing**: Training on anonymized features (PCA-transformed V features)
- **Production-grade inference**: Handling missing values, feature validation
- **Interactive investigation**: Web app for exploring fraud risk factors with SHAP

### Problem Statement

Credit card fraud causes billions in losses annually. Banks need to identify fraudulent transactions quickly to prevent losses while minimizing false alarms that impact legitimate customers. This project builds a model that balances fraud detection rate against false alarm rate.

### Key Features Used

- **Amount**: Transaction amount in dollars (continuous)
- **Time**: Time since first transaction in dataset (continuous)
- **V1-V28**: Principal component analysis features (anonymized transaction details)
  - These represent transformed features protecting customer privacy
  - Each V-feature encodes a combination of merchant, location, behavior patterns
  - PCA-transformed to reduce dimensionality and improve security

### Target Variable

- **Class**: 0 = Legitimate transaction, 1 = Fraudulent transaction
- **Class Distribution**: Highly imbalanced (~99.8% legitimate, 0.2% fraudulent)

### Models Trained

1. **Logistic Regression**: Fast, interpretable baseline for fraud scoring
2. **Random Forest**: Ensemble approach capturing complex fraud patterns

## Tech Stack

- Python 3
- Pandas and NumPy
- Scikit-learn
- imbalanced-learn
- Joblib

## Dataset
Place the CSV file here:

- `data/creditcard.csv`

Target column support:

- `Class`
- `class`
- `fraud`
- `is_fraud`
- `target`

## Installation

### Development
```bash
pip install -r requirements.txt
# or with pinned versions
pip install -r requirements-lock.txt
```

### Production (via pip)
```bash
pip install -e .
```

This registers the CLI command `train-fraud` globally.

## Training

```bash
# Using the CLI (after install -e .)
train-fraud --data data/creditcard.csv --out artifacts

# Or directly
python main.py --data data/creditcard.csv --out artifacts
```

Outputs:
- `artifacts/model.joblib` – trained classifier with preprocessor
- `artifacts/metrics.txt` – evaluation metrics (precision, recall, F1, ROC-AUC)

## Inference

### Batch Prediction
```bash
python inference.py --model artifacts/model.joblib --input new_transactions.csv --output predictions.json
```

### Programmatic Usage
```python
from inference import load_model, predict
import pandas as pd

model_artifact = load_model("artifacts/model.joblib")
df = pd.read_csv("new_transactions.csv")
result = predict(model_artifact, df)
print(result)  # {"predictions": [...], "fraud_probability": [...], "fraud_rate": ...}
```

## Interactive Web App (Streamlit)

### Launch the Web App
```bash
# Install Streamlit and SHAP first
pip install streamlit shap

# Run the app
streamlit run app.py
```

The app provides:
- **Interactive predictions**: Input transaction details and get fraud risk assessment
- **SHAP explanations**: Identify which features drove fraud detection
- **Risk levels**: Color-coded alerts (low, medium, high risk)
- **Dataset overview**: Explore fraud statistics and transaction patterns
- **Real-time scoring**: Fraud probability and confidence metrics

**Browser**: Opens automatically at `http://localhost:8501`

## Jupyter Notebooks

Explore the analysis and training workflow:

```bash
jupyter notebook notebooks/
```

**Available notebooks:**
- `01_eda.ipynb` - Exploratory Data Analysis (class imbalance, amount patterns, feature correlations)
- Add more notebooks for detailed analysis

## Docker

```bash
# Build
docker build -t fraud-detector:latest .

# Train in container
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts fraud-detector:latest

# Predict
docker run --rm -v $(pwd):/app fraud-detector:latest python inference.py --input /app/new_transactions.csv
```

## Testing

```bash
python -m unittest discover -v tests
```

## Production Guide

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)
- Git

### Local Setup

```bash
# Clone the repo
git clone <repo-url>
cd Portfolio-Task-5-Fraud-Production-Style

# Install dependencies (locked versions recommended for production)
pip install -r requirements-lock.txt
# or
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Production Monitoring

**Key Metrics:**
- **Precision**: Must stay high to minimize false positives
- **Recall**: Must stay high to catch actual fraud
- **ROC-AUC**: Overall discriminative power (see artifacts/metrics.txt)
- **Fraud Detection Rate**: % of flagged transactions
- **False Positive Rate**: % of legitimate transactions flagged
- **Inference Latency**: Must be fast for real-time blocking

**Alerts:**
- Set alert if Precision drops below 50% (too many false alarms)
- Set alert if Recall drops below 80% (missing actual fraud)
- Set alert if > 10% of requests fail validation
- Alert if average latency exceeds 100ms per transaction

**Logging:**
All inference runs log detailed fraud statistics:
```
2026-05-26 10:30:45 - INFO - Input validation passed: 10000 transactions
2026-05-26 10:30:46 - INFO - Loaded model from artifacts/model.joblib
2026-05-26 10:30:47 - INFO - Generated predictions for 10000 samples; fraud detected in 125 samples
```

### CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml` runs:
- Dependency installation
- Unit tests (`python -m unittest discover`)
- On every push/PR

### Model Retraining

To retrain with new data:
```bash
python main.py --data data/creditcard_updated.csv --out artifacts
```

New artifacts will overwrite previous `model.joblib` and `metrics.txt`.

### Maintenance

- Keep `requirements-lock.txt` synchronized with production environment
- Test any dependency upgrades in staging first
- Monitor for emerging fraud patterns (data drift detection)
- Archive old models before retraining
- Review flagged transactions periodically for false positives

## Project Flow

1. Load the transaction dataset.
2. Split the data with stratification.
3. Build preprocessing for numeric and categorical features.
4. Apply SMOTE to the training fold to balance the minority class.
5. Compare candidate classifiers (Logistic Regression, Random Forest) using F1-driven cross-validation.
6. Save the best pipeline and the final evaluation summary.

## 📊 Limitations

### Current Constraints

1. **Anonymized Features**: V1-V28 are PCA-transformed, making business interpretation difficult
2. **No Timestamp Information**: Time feature is normalized; actual date/time patterns lost
3. **No Merchant Information**: Merchant category, location, industry details anonymized for privacy
4. **No Customer Context**: No customer age, account history, spending patterns, or geography
5. **SMOTE Artifacts**: Synthetic minority samples may not reflect realistic fraud patterns
6. **Dataset Age**: Credit card fraud patterns evolve; older data may not reflect current tactics
7. **No Real-World Costs**: Model treats false positives and false negatives equally; actual costs differ

### Model Trade-offs

- **Precision vs. Recall Tradeoff**: 
  - High recall catches more fraud but creates customer friction (blocked transactions)
  - High precision reduces false alarms but misses real fraud
- **Interpretability Gap**: Random Forest F1 scores better than Logistic Regression but less explainable
- **SMOTE Complexity**: Improves imbalance but adds training time and potential overfitting risk

## 🚀 Future Scope

### Short-term Enhancements (Weeks)

1. **Threshold Optimization**: Find best decision threshold balancing precision/recall
2. **ROC-AUC Analysis**: Visualize performance across all classification thresholds
3. **Fraud Propensity Scoring**: Convert predictions to risk scores (0-100) instead of binary
4. **Temporal Patterns**: Add time-based features (time of day, day of week, seasonal patterns)
5. **Feature Importance**: Analyze which V-features most strongly indicate fraud
6. **Cost-Based Learning**: Implement custom loss functions reflecting real fraud investigation costs

### Medium-term Extensions (Months)

1. **Real-Time Detection API**: Deploy FastAPI endpoint for immediate transaction flagging
2. **Time-Series Modeling**: Capture sequential fraud patterns (velocity checks, card activation fraud)
3. **Network Analysis**: Detect fraud rings by analyzing merchant-cardholder networks
4. **GRU/LSTM Models**: Use sequence models to learn temporal fraud patterns
5. **Ensemble Methods**: Combine Logistic Regression + Random Forest with voting/stacking
6. **Anomaly Detection**: Unsupervised learning to identify novel fraud patterns
7. **Dashboard**: Build Streamlit fraud monitoring dashboard with recent alerts

### Long-term Vision (Years)

1. **Explainable AI**: Reverse PCA to interpret V-features back to merchant/location patterns
2. **Graph Neural Networks**: Model transaction network structure to detect fraud rings
3. **Transfer Learning**: Pre-train on public fraud datasets, fine-tune on bank-specific data
4. **Behavioral Biometrics**: Incorporate typing patterns, device fingerprints, user behavior
5. **Active Learning**: Iteratively collect hard-to-classify transactions for model improvement
6. **Causal Discovery**: Understand which merchant properties actually cause fraud vs. correlation
7. **Federated Learning**: Train on decentralized bank data without sharing sensitive information

### Research Opportunities

- Compare fraud detection across different transaction types (online, in-store, international)
- Analyze how fraud tactics change over time (concept drift in fraud patterns)
- Study false positive impact on customer satisfaction and retention
- Investigate whether different customer segments have different fraud risk profiles
- Evaluate cost-benefit of different decision thresholds for actual bank operations
- Design real-time fraud prevention systems with feedback loops

## Project Structure

```
.
├── main.py              # Training script with SMOTE
├── inference.py         # Inference with validation
├── app.py               # Streamlit interactive web app
├── pyproject.toml       # Package metadata
├── Dockerfile           # Container definition
├── Makefile             # Convenience commands
├── requirements.txt     # Python dependencies
├── requirements-lock.txt # Pinned versions (includes imbalanced-learn)
├── LICENSE              # MIT
├── README.md            # This file
├── .gitignore           # Git ignore patterns
├── .github/workflows/ci.yml # GitHub Actions
├── data/
│   └── creditcard.csv   # Training dataset
├── notebooks/           # Jupyter notebooks
│   └── 01_eda.ipynb     # Exploratory Data Analysis
├── artifacts/           # Trained model & metrics
└── tests/
    └── test_quick.py    # Smoke tests
```

## Tech Stack

- **Python 3.11+**
- **Pandas & NumPy** – data manipulation
- **Scikit-learn** – ML models and preprocessing
- **imbalanced-learn** – SMOTE for class balancing
- **Joblib** – model persistence
- **Matplotlib & Seaborn** – visualization
