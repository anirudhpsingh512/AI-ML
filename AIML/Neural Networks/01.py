import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top','trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

#print(train_labels[0])

#print(train_images[7])
# plt.imshow(train_images[7], cmap = plt.cm.binary)
# plt.show()

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(train_images, train_labels, epochs = 5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Accuracy : ", test_acc)

# model.save('model2.h5')

model = load_model('model.h5')

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)  #cmap=plt.cm.binary - gives gray scale
    plt.xlabel("Actual:" + class_names[test_labels[i]])
    plt.title("Prediction: "+  class_names[np.argmax(prediction[i])])
    plt.show()