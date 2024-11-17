import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import re
from sklearn.neighbors import KNeighborsClassifier

def normalize_text(text):
    text = re.sub(r'\d+', 'NUMBER', text)
    text = text.lower()
    return text
def clean_text(text):
    # Remove all digits
    text = re.sub(r'\d+', '', text)
    # Remove special characters (optional, depending on your data)
    text = re.sub(r'\W+', ' ', text)
    return text


#df = pd.read_csv("full_dataframe.csv")
#df["Response"] = df['canonical_solution']
#df.drop(['canonical_solution'])

# Example DataFrame with text and boolean labels
data = pd.read_csv("3.5Turbo_dataframe.csv")
#data = pd.concat([data, df], ignore_index=False)
data['Response'] = data['Response'].apply(lambda x: x[50:-30])
data['Response'] = data['Response'].apply(clean_text)
data['Response'] = data['Response'].apply(normalize_text)

print(data)

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['If_human'])

# Separate features (text) and target (label)
X = data['Response']
y = data['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text data to numeric form using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train a decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train_tfidf, y_train)

# Make predictions
predictions = model.predict(X_test_tfidf)

"""
#importances = model.feature_importances
# Get the feature importances
importances = model.feature_importances_

# Get the feature names from the TF-IDF vectorizer
feature_names = vectorizer.get_feature_names_out()

# Combine feature names and their importance scores
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Display the top 10 features
top_features = feature_importance_df.head(15)
print("Top 10 important words/features in the model:")
print(top_features)
"""
# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)



print(f'Accuracy: {accuracy:.2f}')
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC Score: {roc_auc}")

