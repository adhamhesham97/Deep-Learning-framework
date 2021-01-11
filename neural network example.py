#%% load dataset

import numpy as np
import DL

batch_size = 20
num_epochs = 50
samples_per_class = 100
num_classes = 10
hidden_units = 100


# change the directory
# Label_Train, Features_Train, Label_Test, Features_Test = DL.ReadFile("H:\\4th comp\\NN\\MNISTcsv")

#%% training

# choose only 100 examples

num_of_examples = 100

'''
Label_Train_reduced, Features_Train_reduced = Label_Train[:num_of_examples], Features_Train[:,:num_of_examples]

Label_Train_reduced = np.load('y_train_reduced.npy')
Features_Train_reduced = np.load('x_train_reduced.npy')
num_of_features = Features_Train_reduced.shape[0]
'''

Label_Train_reduced = np.load('H:/4th comp/NN/y_train_reduced.npy')
Features_Train_reduced = np.load('H:/4th comp/NN/x_train_reduced.npy')
num_of_features = Features_Train_reduced.shape[0]

# change label from scaler value to vector( 2---> [0, 0, 1, 0, 0, ...] )
L=np.zeros((num_classes, num_of_examples))
for num in Label_Train_reduced:
    L [int(Label_Train_reduced[int(num)])] [int(num)] = 1
Label_Train_reduced_hotone = L


model = DL.model()
model.add(num_of_features, hidden_units, 'Relu')
model.add(hidden_units, num_classes, 'Sigmoid')
optim = DL.optimizer(0.001)
loss_fn = DL.loss_Function('CROSSENTROPY')

model.fit(Features_Train_reduced, Label_Train_reduced_hotone,
          100, num_epochs, optim, loss_fn)


# test on the same trained data set
predicted_labels = np.argmax(model.predict(Features_Train_reduced),axis=0)
accuracy = DL.accuracy(predicted_labels, Label_Train_reduced)
print("Model Accuracy = {}%".format(int(accuracy*100)))



# all examples
'''
num_of_features = Features_Train.shape[0]
num_of_examples = Features_Train.shape[1]
# change label from scaler value to vector( 2---> [0, 0, 1, 0, 0, ...] ) (hot one)
L=np.zeros((num_classes, num_of_examples))
for num in Label_Train:
    L [int(Label_Train[int(num)])] [int(num)] = 1
Label_Train_hotone = L


model = DL.model()
model.add(num_of_features, hidden_units, 'Relu')
model.add(hidden_units, num_classes, 'Relu')
optim = DL.optimizer(1)
loss_fn = DL.loss_Function('MEAN')

model.fit(Features_Train, Label_Train_hotone,
          100, num_epochs, optim, loss_fn)


# test on the same trained data set
predicted_labels = np.argmax(model.predict(Features_Train),axis=0)
accuracy = DL.accuracy(predicted_labels, Label_Train)
print("Accuracy of training dataset = {}%".format(accuracy*100))

# test on the test data set
predicted_labels = np.argmax(model.predict(Features_Test), axis=0)
accuracy = DL.accuracy(predicted_labels, Label_Test)
print("Model Accuracy = {}%".format(accuracy*100))
'''