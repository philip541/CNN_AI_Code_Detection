import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, f1_score, roc_auc_score
import numpy as np 
import re
import pandas as pd


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
    #lambda row: row['canonical_solution'][15:-10] if row['if_human'] == 0 else row['canonical_solution'],
    #axis=1)
df['canonical_solution'] = df['canonical_solution'].apply(lambda x: x[15:-10])


print(df.tail())


y = df["if_human"]
X = df["canonical_solution"]

test_acc_list = []
fscore_list = []
precision_list = []
recall_list = []
AUC_list = []




for i in range(10):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
	X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5)

	def vect(train, test, val, bigram):
		#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
		#Implementing a bigram bag of words model
		if bigram == True:
			bigram = (2,2)
		else:
			bigram = (1,1)
		#Calling the library
		vectorizer = CountVectorizer()
		#Fitting and transforming the vectorizer with the training data, then transorming it with the testing data
		xtrain = vectorizer.fit_transform(train)
		xtest = vectorizer.transform(test)
		v = vectorizer.transform(val)

		vocab = vectorizer.vocabulary_

		return xtrain, xtest, v, list(vocab.items())




	def lSVC(xtrain, ytrain, xval, yval, xtest, ytest, vocab):
		#Keeping track of the optimal c value and the corresponding f score
		opt_c = [0, 0]
		#Looping through 1 x 10^-10 to 1 x 10^10 -- To try different c values
		"""
		for exp in range(-10, 10):
			c = 10 * 10**exp
			#Creating the model with the current c value, had to raise iterations to limit failed to converge
						#I set dual to true because otherwise it gives a warnign message saying something like "Duals defualt is changing in 2025"
			model = LinearSVC(C=c, dual=True, max_iter=2000)
			#Fitting the model with the training data
			model.fit(xtrain, ytrain)
			#Using the validation data to predict, then calculating the fscore based on those predictions
			y_pred = model.predict(xval)
			fscore = f1_score(yval, y_pred)
			#Checking of the current f score is better then the one in the optimal c score list
			if fscore > opt_c[1]:
				#If it is, we set the first item to the c value and the second item to the fscore
				opt_c[0] = 10 * 10**exp
				opt_c[1] = fscore
		"""
		#After determing the optimal c score, we initate the model one last time with the best c score
		final_model = LinearSVC(C=.1, dual=True, max_iter=2000)
		#Fit on the train data again
		final_model.fit(xtrain, ytrain)

		#Here we get the coefficients from the model, and have the vocabulary from the vectorizer
		coefficients = final_model.coef_[0]
		#print(coefficients)
	#https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
		#Arg sort creates a list of indexes in order of the values size, documentation above
		indices = np.argsort(coefficients)
		#print(indexes)

		print("Below are the most Human-like words")
		for idx in indices[:7]:
			print(vocab[idx])
	    
		print("Below are the most AI-like words")
		for idx in indices[-7:]:
			print(vocab[idx])


		#I put some of the these words in my journal




		#Finally, we use the model to predict the y test values with the x test values, and calculate the fscore as well as accuracy
		pred = final_model.predict(xtest)
		fscore = f1_score(ytest, pred)
		accuracy = accuracy_score(ytest, pred)
		precision = precision_score(ytest, pred)
		recall = recall_score(ytest, pred)
		ROC_AUC = roc_auc_score(ytest, pred)



		#Returning the final f score
		return fscore, accuracy, precision, recall, ROC_AUC





	tr, te, v, vocab = vect(X_train, X_test, X_val, False)
	f1score, accuracy, precision, recall, ROC_AUC = lSVC(tr, y_train, v, y_val, te, y_test, vocab)

	print(f"Test Accuracy: {accuracy}")
	print(f"F1 Score: {f1score}")
	print(f"Precision: {precision}")
	print(f"Recall: {recall}")
	print(f"ROC AUC Score: {ROC_AUC}")

	test_acc_list.append(accuracy)
	fscore_list.append(f1score)
	precision_list.append(precision)
	recall_list.append(recall)
	AUC_list.append(ROC_AUC)



print((sum(test_acc_list) / len(test_acc_list)))
print((sum(fscore_list) / len(fscore_list)))
print((sum(precision_list) / len(precision_list)))
print((sum(recall_list) / len(recall_list)))
print((sum(AUC_list) / len(AUC_list)))





