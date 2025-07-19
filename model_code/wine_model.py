import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import dagshub
import requests
import mlflow
import streamlit as st
import numpy as np
import joblib

# Load the wine quality dataset
data = pd.read_csv(r'C:\Users\USER\Documents\skillfyme_mlops\Flask_Model\data\winequality-red.csv')

# Features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create and train the classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score (weighted): {f1:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")

dagshub.init(repo_owner='vinayswaroop699', repo_name='FLASK_MODEL', mlflow=True)

with mlflow.start_run():
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.5)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score_weighted", f1)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    # Save the trained model
    joblib.dump(clf, "wine_quality_rf.pkl")

# Streamlit app
st.title("Wine Quality Prediction")

st.write("Enter the wine characteristics to predict its quality:")

feature_names = X.columns.tolist()
user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=float(X[feature].mean()))
    user_input.append(val)

if st.button("Predict Quality"):
    model = joblib.load("wine_quality_rf.pkl")
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
