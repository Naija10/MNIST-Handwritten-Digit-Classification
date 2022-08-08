from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation 							
import numpy as np
import cv2

mnist = keras.datasets.mnist
(train_inputs,train_targets),(test_inputs,test_targets) = mnist.load_data() 
normalised_train_inputs = train_inputs/255
normalised_test_inputs=test_inputs/255

model = keras.models.load_model("Models/model.hdf5")

predictions=model.predict(normalised_test_inputs)
for i in range(len(normalised_test_inputs)):
     print(np.argmax(np.round(predictions[i])))




