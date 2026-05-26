"""
Fraud detection model inference with schema validation and logging.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_input(df: pd.DataFrame) -> bool:
    """Validate input DataFrame for fraud detection."""
    if df.empty:
        logger.error("Input DataFrame is empty")
        return False
    logger.info(f"Input validation passed: {len(df)} transactions")
    return True


def load_model(model_path: Path):
    """Load trained model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    artifact = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return artifact


def predict(model_artifact, df: pd.DataFrame) -> Dict[str, Any]:
    """Generate predictions on input DataFrame."""
    # Handle both dict (preprocessor + model) and direct model
    if isinstance(model_artifact, dict):
        preprocessor = model_artifact["preprocessor"]
        model = model_artifact["model"]
        X = preprocessor.transform(df)
    else:
        X = df
        model = model_artifact
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    logger.info(f"Generated predictions for {len(predictions)} samples; fraud detected in {predictions.sum()} samples")
    return {
        "predictions": predictions.tolist(),
        "fraud_probability": probabilities.tolist(),
        "fraud_count": int(predictions.sum()),
        "fraud_rate": float(predictions.sum() / len(predictions))
    }


def main():
    """CLI entrypoint for inference."""
    import argparse
    parser = argparse.ArgumentParser(description="Credit card fraud detection inference")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    if not validate_input(df):
        raise ValueError("Input validation failed")
    
    model_artifact = load_model(args.model)
    result = predict(model_artifact, df)
    
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
