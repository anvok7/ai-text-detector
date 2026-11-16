# AI-Generated Text Detector

A machine learning-based NLP application that estimates whether a given piece of text is **human-written or AI-generated**.

The project uses the **HC3 (Human-ChatGPT Comparison Corpus)** dataset and combines **word-level and character-level TF-IDF features** with **Logistic Regression** for classification.

The trained model is deployed through a simple **Streamlit web application** that displays the predicted class and probability breakdown.

---

## Project Overview

AI-generated text has become increasingly difficult to distinguish from human-written content. This project explores a traditional machine learning approach for detecting AI-generated text using linguistic and character-level patterns.

The system:

1. Loads the HC3 dataset.
2. Converts the original question-answer structure into individual text samples.
3. Labels human-written and AI-generated responses.
4. Extracts word-level and character-level TF-IDF features.
5. Trains a Logistic Regression classifier.
6. Evaluates the model on a stratified held-out test set.
7. Saves the trained model for later inference.
8. Provides a Streamlit interface for testing new text.

---

## Features

- Human vs AI-generated text classification
- Word-level TF-IDF features
- Character-level TF-IDF features
- Logistic Regression classifier
- Stratified train/test split
- Accuracy, precision, recall and F1-score evaluation
- Confusion matrix visualization
- Probability-based predictions
- Interactive Streamlit web interface
- Saved model using Joblib

---

## System Architecture

```text
                    HC3 Dataset
                         |
                         v
                  Data Preparation
                         |
                         v
              Cleaned & Labeled Text
                         |
                         v
              Train / Test Split
                         |
              +----------+----------+
              |                     |
              v                     v
       Word-level TF-IDF     Character-level TF-IDF
              |                     |
              +----------+----------+
                         |
                         v
                 Feature Union
                         |
                         v
              Logistic Regression
                         |
                         v
                  Trained Model
                         |
              +----------+----------+
              |                     |
              v                     v
        Model Evaluation       Streamlit App
                                    |
                                    v
                         Human / AI Prediction
                         + Probability Scores
