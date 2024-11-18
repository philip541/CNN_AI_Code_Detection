import os
import math
import numpy as np
import matplotlib.pyplot as plt



arrays = []
file_lens = []
dimension = 34




#Loading binary and converting it to numpy arrays
def file_to_array(directory):
	arrays = []
	file_lens = []
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


folders = ["Test/Human", "Train/Human", "Test/AI", "Train/AI"]
file_names = ["/Human_image", "/Human_image", "/AI_image", "/AI_image"]

for i in range(4):
	path = "/Users/philipnegrin/Downloads/AICodeDetection/Code_Net/Data_B4/" + folders[i]
	arrays_ai, file_lengths_AI = file_to_array(path)
	counter = 0
	for array in arrays_ai:
		print(counter)
		counter += 1
		plt.imshow(array, cmap="gray")
		path = "/Users/philipnegrin/Downloads/AICodeDetection/Code_Net/Data_GS4/" + folders[i] + file_names[i] + str(counter)
		plt.savefig(path)
		plt.close()



























