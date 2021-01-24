import numpy as np
import DL

# change the directory
Label_Train, Features_Train, Label_Test, Features_Test = DL.ReadFile("F:\\eural\\project2\\Deep-Learning-framework-main\\MNISTcsv")
print(Features_Train.shape)

Features_Train_flattened=Features_Train.reshape(60000,28*28)
Features_Test_flattened=Features_Test.reshape(10000,28*28)
pca=DL.PCA(0.98)
pca.fit(Features_Train_flattened)
Features_Train_reduced = pca.transform(Features_Train_flattened)
print(Features_Train_reduced.shape)


#%% training

batch_size = 64
num_epochs = 10
num_classes = 10
hidden_units = 300

Label_Train_hotone = DL.hot_one(Label_Train, num_classes)
input_dimensions=Features_Train_reduced.shape[0]
print(input_dimensions)
model = DL.model()
model.input_dims(input_dimensions)
model.add('Relu', hidden_units)
model.add('Linear', num_classes)
optim = DL.optimizer('gd',0.2,0.2)
loss_fn = DL.loss_Function('SoftmaxCrossEntropy')
loss_fn.setLambda(0)
print('Features_Train_reduced',Features_Train_reduced.shape)
model.fit(Features_Train_reduced, Label_Train_hotone,batch_size, num_epochs, optim, loss_fn)


#%% testing

# test on the same trained data
print((Features_Train_reduced).shape)
predicted_labels = np.argmax(model.predict((Features_Train_reduced)), axis=0)
accuracy = DL.accuracy(predicted_labels, Label_Train)
print("Accuracy of training dataset = {:.2f}%".format(accuracy*100))

# test on the test data set
Features_Test_reduced = pca.transform(Features_Test_flattened)
predicted_labels = np.argmax(model.predict(Features_Test_reduced), axis=0)
accuracy = DL.accuracy(predicted_labels, Label_Test)
print("Model Accuracy = {:.2f}%".format(accuracy*100))


#%% store and load model

# DL.store(model, "FC MNIST model") # store
# model = DL.load("FC MNIST model") # load

