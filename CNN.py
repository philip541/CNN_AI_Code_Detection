import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tensorflow.keras.optimizers import Adam


accuracy_list = []
precision_list = []
recall_list = []
fscore_list = []
AUC_list = []



AI_train_path = ""
AI_test_path = ""

Human_train_path = ""
Human_test_path = ""

train_path = "/Users/philipnegrin/Downloads/AICodeDetection/ShortResponseData/Data/Train"
test_path = "/Users/philipnegrin/Downloads/AICodeDetection/ShortResponseData/Data/Test"

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

#Default is .001
#learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
learning_rates = [0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013]
learningrate = .001
#batch_sizes = [8, 16, 32, 64, 128, 256, 512]
batchsize = 32
"""
Epochs
Activation functions
Compiling functions instead of adam
Loss functions
"""

#for batchsize in batch_sizes:
for i in range(1):
    train_generator = train_datagen.flow_from_directory(train_path, color_mode="grayscale" ,class_mode="binary")
    test_generator = train_datagen.flow_from_directory(test_path, color_mode="grayscale" ,class_mode="binary")


    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Updated for binary classification

    # Compile the model with Binary Crossentropy loss
    optimizer = Adam(learning_rate=learningrate)
    #Default is .001, 32
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(train_generator, epochs=6, 
                        validation_data=test_generator
                        ,batch_size=batchsize
                        )


    #plt.plot(history.history['accuracy'], label='accuracy')
    #plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.ylim([0.5, 1])
    #plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_generator, verbose=2)

    print(test_acc)

    predictions = model.predict(test_generator)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    predicted_probabilities = predictions.flatten()

    true_classes = test_generator.classes

    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    precision = precision_score(true_classes, predicted_classes)
    recall = recall_score(true_classes, predicted_classes)
    roc_auc = roc_auc_score(true_classes, predicted_probabilities)

    print(f"Test Accuracy: {test_acc}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"ROC AUC Score: {roc_auc}")


    accuracy_list.append(test_acc)
    fscore_list.append(f1)
    precision_list.append(precision)
    recall_list.append(recall)
    AUC_list.append(roc_auc)



#plt.show()


print(accuracy_list)
print(fscore_list)
print(precision_list)
print(recall_list)
print(AUC_list)

"""
print((sum(accuracy_list) / len(accuracy_list)))
print((sum(fscore_list) / len(fscore_list)))
print((sum(precision_list) / len(precision_list)))
print((sum(recall_list) / len(recall_list)))
print((sum(AUC_list) / len(AUC_list)))
"""

"""
4-[0.7857142686843872, 0.738095223903656, 0.8095238208770752, 0.75, 0.726190447807312, 0.738095223903656, 0.6190476417541504]
6-[0.8333333134651184, 0.7857142686843872, 0.761904776096344, 0.761904776096344, 0.773809552192688, 0.8333333134651184, 0.7857142686843872]
8-[0.7976190447807312, 0.75, 0.75, 0.8214285969734192, 0.738095223903656, 0.8333333134651184, 0.8095238208770752]
10-[0.8452380895614624, 0.8214285969734192, 0.8333333134651184, 0.738095223903656, 0.773809552192688, 0.8214285969734192, 0.8452380895614624]
11-[0.8095238208770752, 0.8214285969734192, 0.7857142686843872, 0.8095238208770752, 0.8571428656578064, 0.8095238208770752, 0.8333333134651184]
12-[0.8214285969734192, 0.8333333134651184, 0.8214285969734192, 0.726190447807312, 0.7976190447807312, 0.8333333134651184, 0.7857142686843872]
"""
"""
11-[0.7023809552192688, 0.8452380895614624, 0.773809552192688, 0.8095238208770752, 0.7023809552192688, 0.8095238208770752, 0.761904776096344]
10-[0.8571428656578064, 0.8571428656578064, 0.761904776096344, 0.8333333134651184, 0.7857142686843872, 0.8095238208770752, 0.8333333134651184]
"""

