#%% load dataset

import numpy as np
import DL

# change the directory
Label_Train, Features_Train, Label_Test, Features_Test = DL.ReadFile("H:\\4th comp\\NN\\cifar-10-batches-py")

# tronsform to dimensions (m, h, w, c)
Features_Train = Features_Train.T.reshape(-1,3,32,32).transpose(0, 2, 3, 1)
Features_Test = Features_Test.T.reshape(-1,3,32,32).transpose(0, 2, 3, 1)

Features_Train_small = Features_Train[0:10000] #choose the first 10000 examples only
Features_Test_small = Features_Test[0:1000] #choose the first 1000 examples only


#%% training

batch_size = 256
num_epochs = 1
num_classes = 10
hidden_units = 100

input_dimensions = (32, 32, 3)

# change each label from scaler value to vector( 2 ---> [0, 0, 1, 0, 0, ...] ) (hot one)
Label_Train_hotone = DL.hot_one(Label_Train, num_classes)

'''
conv parameters:
Filter_size, num_of_filters, Stride, padding, activation_type

pool parameters:
Filter_size, Stride
'''
model = DL.model()
model.input_dims(input_dimensions)
model.add('conv', (3, 3), 32, 1, 1, "Relu")
model.add('maxpool', (2, 2), 2)
model.add('flatten')
# model.add('Relu', hidden_units)
model.add('Linear', num_classes)
optim = DL.optimizer(0.00001)
loss_fn = DL.loss_Function('SoftmaxCrossEntropy')
loss_fn.setLambda(0)
model.fit(Features_Train_small, Label_Train_hotone,
          batch_size, num_epochs, optim, loss_fn)


#%% testing

# test on the same trained data set
predicted_labels = np.argmax(model.predict(Features_Train_small), axis=0)
accuracy = DL.accuracy(predicted_labels, Label_Train)
print("Accuracy of training dataset = {:.2f}%".format(accuracy*100))

# test on the test data set
predicted_labels = np.argmax(model.predict(Features_Test_small), axis=0)
accuracy = DL.accuracy(predicted_labels, Label_Test)
print("Model Accuracy = {:.2f}%".format(accuracy*100))


#%% store and load model

# DL.store(model, "CNN CIFAR model") # store

# model = DL.load("CNN CIFAR model") # load
