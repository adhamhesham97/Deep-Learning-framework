#%% load dataset

import numpy as np
import DL

# change the directory
Label_Train, Features_Train, Label_Test, Features_Test = DL.ReadFile("H:\\4th comp\\NN\\MNISTcsv")

#%% training

batch_size = 25
num_epochs = 10
num_classes = 10
hidden_units = 300
Lambda=0

num_of_features = Features_Train.shape[0]
num_of_examples = Features_Train.shape[1]

# change each label from scaler value to vector( 2 ---> [0, 0, 1, 0, 0, ...] ) (hot one)
Label_Train_hotone = DL.hot_one(Label_Train, num_classes)

model = DL.model()
model.add(num_of_features, hidden_units, 'Relu')
model.add(hidden_units, num_classes, 'Linear')
optim = DL.optimizer(0.001)
loss_fn = DL.loss_Function('SoftmaxCrossEntropy')

model.fit(Features_Train, Label_Train_hotone,
          batch_size, num_epochs, Lambda, optim, loss_fn)


# test on the same trained data set
predicted_labels = np.argmax(model.predict(Features_Train), axis=0)
accuracy = DL.accuracy(predicted_labels, Label_Train)
print("Accuracy of training dataset = {:.2f}%".format(accuracy*100))

# test on the test data set
predicted_labels = np.argmax(model.predict(Features_Test), axis=0)
accuracy = DL.accuracy(predicted_labels, Label_Test)
print("Model Accuracy = {:.2f}%".format(accuracy*100))
