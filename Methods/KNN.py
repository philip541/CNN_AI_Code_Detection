import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report
)
import matplotlib.pyplot as plt

# Sample data (replace with your own DataFrame)
# Assume `df` is your DataFrame with a 'text' column for features and 'label' column for target
# df = pd.DataFrame({'text': [...], 'label': [...]})

df = pd.read_csv("3.5Turbo_dataframe.csv")
df['Response'] = df['Response'].apply(lambda x: x[30:-20])

# Split data into training and testing sets
X = df['Response']       # Text data
y = df['If_human']      # Binary labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Transform the text data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the KNeighborsClassifier (adjust `n_neighbors` as needed)
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = knn.predict(X_test_tfidf)
y_pred_proba = knn.predict_proba(X_test_tfidf)[:, 1]  # Probability scores for positive class

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.10f}")
print(f"Precision: {precision:.10f}")
print(f"Recall: {recall:.10f}")
print(f"F1 Score: {f1:.10f}")
print(f"AUC-ROC: {auc_roc:.10f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot AUC-ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
