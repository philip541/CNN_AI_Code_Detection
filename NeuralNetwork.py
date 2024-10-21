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

accuracy_list = []
precision_list = []
recall_list = []
fscore_list = []
AUC_list = []




df = pd.read_csv("full_dataframe.csv")

def normalize_text(text):
    text = re.sub(r'\d+', 'NUMBER', text)
    text = text.lower()
    return text
def clean_text(text):
	# Remove all digits
	text = re.sub(r'\d+', '', text)
	# Remove special characters (optional, depending on your data)
	#text = re.sub(r'\W+', ' ', text)
	return text



df['canonical_solution'] = df['canonical_solution'].apply(clean_text)
df['canonical_solution'] = df['canonical_solution'].apply(normalize_text)
#df['canonical_solution'] = df.apply(
 #   lambda row: row['canonical_solution'][15:-10] if row['if_human'] == 0 else row['canonical_solution'],
  #  axis=1)
df['canonical_solution'] = df['canonical_solution'].apply(lambda x: x[15:-10])


# Encode the labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['if_human'])

for i in range(10):
	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(df['canonical_solution'], df['label'], test_size=0.2, random_state=42)

	# Tokenize the text
	tokenizer = Tokenizer(num_words=5000)
	tokenizer.fit_on_texts(X_train)

	X_train = tokenizer.texts_to_sequences(X_train)
	X_test = tokenizer.texts_to_sequences(X_test)

	# Pad the sequences to ensure uniform input length
	maxlen = 100
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
	history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)

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


	accuracy_list.append(accuracy)
	fscore_list.append(f1)
	precision_list.append(precision)
	recall_list.append(recall)
	AUC_list.append(roc_auc)




print((sum(accuracy_list) / len(accuracy_list)))
print((sum(fscore_list) / len(fscore_list)))
print((sum(precision_list) / len(precision_list)))
print((sum(recall_list) / len(recall_list)))
print((sum(AUC_list) / len(AUC_list)))





















