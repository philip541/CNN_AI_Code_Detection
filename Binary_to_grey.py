import os
import math
import numpy as np
import matplotlib.pyplot as plt


arrays = []
file_lens = []
dimension = 12

#Loading binary and converting it to numpy arrays
def file_to_array(directory):
	for files in os.listdir(directory):
		if str(files) == ".DS_Store":
			pass
		else:
			path = directory + "/" + str(files)
			print(str(files))
			print(path)
			with open(path, "r") as file:
				content = file.read()
				content = content.split(" ")
				content.pop(-1)


				sq = math.sqrt(len(content))  #Square rooting length of files (finding dimensions)
				sq = math.floor(sq)           #Rounding down, didn't want values to be annoying floats
				file_lens.append(int(sq))     #Adding ints of values to a list

				for index in range(len(content)):
					content[index] = int(content[index], 2)


				array = np.zeros((dimension, dimension))
				counter = 0
				for i in range(dimension):
					for f in range(dimension):
						if counter < len(content):
							array[i, f] = content[counter]
						counter += 1

				arrays.append(array)

	return arrays, file_lens



path = "/Users/philipnegrin/Downloads/AICodeDetection/ShortResponseData/Data/Train/AI"
arrays_ai, file_lengths_AI = file_to_array(path)
Counter = 0
for array in arrays_ai:
	print(Counter)
	Counter += 1
	plt.imshow(array, cmap="gray")
	path = "/Users/philipnegrin/Downloads/AICodeDetection/ShortResponseData/Data/Train/AI1" + "/AI_image" + str(Counter)
	plt.savefig(path)







