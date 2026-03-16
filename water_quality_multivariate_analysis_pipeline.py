"""
Water Quality Multivariate Analysis Pipeline

Description
-----------
End-to-end data science pipeline to analyze water quality indicators
and predict compliance risks using machine learning.

Pipeline Steps
--------------
1. Load laboratory data
2. Perform exploratory analysis
3. Train predictive regression model
4. Evaluate model performance
5. Export predictions and metrics

This project demonstrates a real-world application of data science
for environmental monitoring and operational decision support.
"""

import time
import datetime as dt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------
# Utility functions
# ---------------------------------------------------

def current_time():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------
# Simulated dataset generator
# (for portfolio purposes)
# ---------------------------------------------------

def generate_sample_data(n=500):

    np.random.seed(42)

    data = pd.DataFrame({

        "ph": np.random.normal(7, 0.5, n),
        "turbidity": np.random.normal(3, 1, n),
        "chlorine": np.random.normal(1.5, 0.3, n),
        "conductivity": np.random.normal(250, 40, n),
        "temperature": np.random.normal(22, 2, n)

    })

    # Target variable (water quality score)
    data["quality_index"] = (
        0.3 * data["ph"]
        + 0.2 * data["chlorine"]
        - 0.15 * data["turbidity"]
        + 0.001 * data["conductivity"]
        + np.random.normal(0, 0.5, n)
    )

    return data


# ---------------------------------------------------
# Pipeline
# ---------------------------------------------------

def run_pipeline():

    print("Pipeline started:", current_time())

    start_time = time.time()

    # Step 1 — Load data
    print("Generating dataset...")

    df = generate_sample_data()

    print("Dataset shape:", df.shape)

    # Step 2 — Feature selection
    X = df.drop(columns=["quality_index"])
    y = df["quality_index"]

    # Step 3 — Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4 — Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5 — Model training
    print("Training model...")

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Step 6 — Prediction
    predictions = model.predict(X_test_scaled)

    # Step 7 — Metrics
    r2 = r2_score(y_test, predictions)

    # Convert regression to classification for metrics example
    threshold = y_test.median()

    y_test_class = (y_test > threshold).astype(int)
    pred_class = (predictions > threshold).astype(int)

    precision = precision_score(y_test_class, pred_class)
    recall = recall_score(y_test_class, pred_class)
    f1 = f1_score(y_test_class, pred_class)

    print("\nModel Performance")
    print("------------------")
    print("R2:", round(r2, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 Score:", round(f1, 3))

    # Step 8 — Save outputs
    results = pd.DataFrame({
        "actual": y_test,
        "predicted": predictions
    })

    results.to_csv("water_quality_predictions.csv", index=False)

    metrics = pd.DataFrame({
        "R2": [r2],
        "Precision": [precision],
        "Recall": [recall],
        "F1": [f1]
    })

    metrics.to_csv("model_metrics.csv", index=False)

    print("\nOutputs saved:")
    print("water_quality_predictions.csv")
    print("model_metrics.csv")

    end_time = time.time()

    print("\nExecution time (minutes):", (end_time - start_time) / 60)
    print("Pipeline finished:", current_time())


# ---------------------------------------------------
# Run
# ---------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
