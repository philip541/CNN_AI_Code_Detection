import pandas as pd
import re
import os
from BinaryFileConverter import read_file_as_binary, write_binary_to_file


AI_train = "/Users/philipnegrin/Downloads/AICodeDetection/ShortResponseData/Data/Train/AI"
AI_test = "/Users/philipnegrin/Downloads/AICodeDetection/ShortResponseData/Data/Test/AI"

Human_train = "/Users/philipnegrin/Downloads/AICodeDetection/ShortResponseData/Data/Train/Human"
Human_test = "/Users/philipnegrin/Downloads/AICodeDetection/ShortResponseData/Data/Test/Human"


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
df['canonical_solution'] = df['canonical_solution'].apply(lambda x: x[15:-10])


"""
328

123
41
123
41
"""

human_df = df[df['if_human'] == 1]
AI_df = df[df['if_human'] == 0]


counter = 0
for index, row in human_df.iterrows():
	counter += 1
	if counter < 123:
		path = Human_train + "/HumanBinary" + str(counter)
		with open(path, "w") as file:
			file.write(row['canonical_solution'])

		binary_content = read_file_as_binary(path)
		write_binary_to_file(binary_content, path)

	else:
		path = Human_test + "/HumanBinary" + str(counter)
		with open(path, "w") as file:
			file.write(row['canonical_solution'])

		binary_content = read_file_as_binary(path)
		write_binary_to_file(binary_content, path)


counter = 0
for index, row in AI_df.iterrows():
	counter += 1
	if counter < 123:
		path = AI_train + "/AIBinary" + str(counter)
		with open(path, "w") as file:
			file.write(row['canonical_solution'])

		binary_content = read_file_as_binary(path)
		write_binary_to_file(binary_content, path)

	else:
		path = AI_test + "/AIBinary" + str(counter)
		with open(path, "w") as file:
			file.write(row['canonical_solution'])

		binary_content = read_file_as_binary(path)
		write_binary_to_file(binary_content, path)














