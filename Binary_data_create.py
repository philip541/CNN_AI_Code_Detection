import pandas as pd
import re
import os
from BinaryFileConverter import read_file_as_binary, write_binary_to_file



AI_train_path = "/Users/philipnegrin/Downloads/AICodeDetection/Code_Net/Data_B4/Train/AI"
AI_test_path = "/Users/philipnegrin/Downloads/AICodeDetection/Code_Net/Data_B4/Test/AI"

Human_train_path = "/Users/philipnegrin/Downloads/AICodeDetection/Code_Net/Data_B4/Train/Human"
Human_test_path = "/Users/philipnegrin/Downloads/AICodeDetection/Code_Net/Data_B4/Test/Human"


df = pd.read_csv("3.5Turbo_dataframe.csv")
df['Response'] = df['Response'].apply(lambda x: x[50:-30])

#df['Response'] = df['Response'].apply(lambda x: x[30:])
#df.to_csv("Human_02243_dataframe.csv", index=False)

human_df = df[df['If_human'] == True]
AI_df = df[df['If_human'] == False]

split_point = .75 * human_df.count()
split_point = int(split_point[0])


counter = 0
for index, row in human_df.iterrows():
	counter += 1
	print(counter)
	if counter < split_point:
		path = Human_train_path + "/Human_Binary" + str(counter)

		with open(path, "w") as file:
			file.write(row['Response'])

		binary_content = read_file_as_binary(path)
		write_binary_to_file(binary_content, path)

	else:
		path = Human_test_path + "/Human_Binary" + str(counter)

		with open(path, "w") as file:
			file.write(row['Response'])

		binary_content = read_file_as_binary(path)
		write_binary_to_file(binary_content, path)

counter = 0
for index, row in AI_df.iterrows():
	counter += 1
	print(counter)
	if counter < split_point:
		path = AI_train_path + "/AI_Binary" + str(counter)

		with open(path, "w") as file:
			file.write(row['Response'])

		binary_content = read_file_as_binary(path)
		write_binary_to_file(binary_content, path)

	else:
		path = AI_test_path + "/AI_Binary" + str(counter)

		with open(path, "w") as file:
			file.write(row['Response'])

		binary_content = read_file_as_binary(path)
		write_binary_to_file(binary_content, path)

