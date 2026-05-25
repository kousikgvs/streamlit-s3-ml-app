"""
Train the Loan Approval model and upload artifacts (model + scaler) to S3.

Reads AWS credentials from .env and uses region ap-south-2 (see s3_operations.py).
Run:  python train_and_upload.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from s3_operations import upload_file

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "Loan_Dataset.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "random_forest_model.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")


def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
    df.drop(columns=["Loan_ID"], inplace=True)

    # Fill missing values
    df["Gender"]           = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"]          = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"]       = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"]    = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
    df["LoanAmount"]       = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"]   = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    # Encode categorical columns
    for col in ["Gender", "Married", "Dependents", "Education",
                "Self_Employed", "Property_Area", "Loan_Status"]:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df


def main():
    print(f"Loading dataset: {DATASET_PATH}")
    df = load_and_preprocess(DATASET_PATH)
    print(f"Shape: {df.shape}, Nulls: {df.isnull().sum().sum()}")

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Save locally
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {SCALER_PATH}")

    # Upload to S3
    print("Uploading artifacts to S3...")
    ok_model  = upload_file(MODEL_PATH,  "models")
    ok_scaler = upload_file(SCALER_PATH, "models")

    if ok_model and ok_scaler:
        print("✅ Model and scaler uploaded to S3 successfully.")
    else:
        print("❌ One or more uploads failed. Check logs above.")


if __name__ == "__main__":
    main()
