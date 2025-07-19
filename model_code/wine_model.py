import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import dagshub

# DagsHub experiment tracking
dagshub.init("wine-quality-classification", "vinayswaroop699", "wine-quality-classification")

# Load dataset
df = pd.read_csv(r"C:\Users\USER\Documents\skillfyme_mlops\Flask_Model\data\winequality-red.csv")

# Prepare features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Convert to binary classification (good: quality >= 7)
y = (y >= 7).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Log metrics to DagsHub
with dagshub.dagshub_logger() as logger:
    logger.log_metrics(report["accuracy"], step=0)
    for label in ["0", "1"]:
        logger.log_metrics({
            f"precision_{label}": report[label]["precision"],
            f"recall_{label}": report[label]["recall"],
            f"f1-score_{label}": report[label]["f1-score"]
        }, step=0)