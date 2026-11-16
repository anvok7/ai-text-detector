import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = "data/raw/hc3_processed.csv"
MODEL_PATH = "models/ai_text_detector.joblib"
RESULTS_PATH = "results/metrics.json"


def main():
    print("Loading processed dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df["text"]
    y = df["label"]

    print(f"Total samples: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_features=100000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    print("\nTraining model...")

    pipeline.fit(X_train, y_train)

    print("Training complete.")

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\nModel Performance")
    print("-----------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print("---------------------")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Human", "AI-generated"]
        )
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_samples": len(X_train),
        "testing_samples": len(X_test)
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Human", "AI-generated"],
        yticklabels=["Human", "AI-generated"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("AI Text Detector - Confusion Matrix")

    plt.tight_layout()

    plt.savefig("results/confusion_matrix.png", dpi=300)
    plt.close()

    print("\nModel saved to:", MODEL_PATH)
    print("Metrics saved to:", RESULTS_PATH)
    print("Confusion matrix saved to: results/confusion_matrix.png")


if __name__ == "__main__":
    main()