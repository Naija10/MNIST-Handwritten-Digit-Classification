#importing libraries
from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation,Flatten
import numpy as np
import cv2

#load the dataset and split the model
mnist = keras.datasets.mnist
(train_inputs,train_targets),(test_inputs,test_targets) = mnist.load_data() 
normalised_train_inputs = train_inputs/255
normalised_test_inputs=test_inputs/255

#Defining the model
model=keras.Sequential()
model.add(Flatten(input_shape=normalised_train_inputs.shape[1:]))
model.add(Dense(10,input_shape = normalised_train_inputs.shape[1:]))
model.add(Activation('relu'))
model.add(Dense(10)) 							
model.add(Activation('relu'))
model.add(Dense(10)) 							
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax')) 									
model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.Adam(learning_rate=0.05),metrics=['accuracy'])

#Training the model
model.fit(normalised_train_inputs,train_targets,batch_size=260,epochs=4)

#Saving the model
model.save("Models/model.hdf5")

#testing the accuracy
test_loss,test_acc=model.evaluate(normalised_test_inputs,test_targets)
print('Test loss',test_loss)
print('Test accuracy',test_acc)



