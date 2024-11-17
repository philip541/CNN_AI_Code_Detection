"""
Pre-trained Transformer-based models (CodeBERT, GPT/Codex)
Random Forest or Decision Tree on Token Embeddings
SVM with feature extraction
RNN-based models (LSTM, GRU)
Tree-based models (AST, Tree-LSTMs)
N-Gram Language Models
k-NN on Token Embeddings
Transfer Learning from Pre-trained CNNs
Ensemble Methods
CodeBERT
GPT’s Codex
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import re
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np



df = pd.read_csv("3.5Turbo_dataframe.csv")

df['Response'] = df['Response'].apply(lambda x: x[50:-30])


label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['If_human'])

print(df.head())


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Response'], df['label'], test_size=0.2, random_state=42)

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to ensure uniform input length
maxlen = 200
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Define the model
model = Sequential()
#Adding layers to the model -- I messed around with adding additional layers as well as number of neurons in each layer
model.add(Embedding(input_dim=5000, output_dim=128, input_length=maxlen))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)), #return_sequences=True))
#model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(32, activation='relu'))  # Additional Dense layer
model.add(Dense(1, activation='sigmoid'))


# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=8, batch_size=128, validation_split=0.2, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Accuracy: {accuracy}')

predictions_prob = model.predict(X_test)
predicted_classes = (predictions_prob > 0.5).astype(int).flatten()  # Convert probabilities to binary class labels
#print(predicted_classes)

f1 = f1_score(y_test, predicted_classes)
precision = precision_score(y_test, predicted_classes)
recall = recall_score(y_test, predicted_classes)
roc_auc = roc_auc_score(y_test, predictions_prob)

print(f'Test Accuracy: {accuracy}')
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC Score: {roc_auc}")
























