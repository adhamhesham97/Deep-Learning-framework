# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 23:47:45 2021

@author: Aya El Ashry
"""

import numpy as np
import matplotlib.pyplot as plt
from Layer import layer
from Visualization import visualization

class model:
    
    def __init__(self):
        self.layers=[]
        self.parameters = []

        
    def add(self, in_nodes, out_nodes, activation_type):
        l=layer(in_nodes, out_nodes, activation_type)
        self.layers.append(l)
        self
        
    def Initialize_Parameters(self):        
        for Layer in self.layers:
            w,b=Layer.getParams()
            w= np.random.randn(w.shape[0],w.shape[1]) * 0.01
            b = np.zeros(b.shape)
            Layer.setParams(w, b)

    def create_mini_batches(self, X, y, batch_size): 
        mini_batches = [] 
        n_minibatches = X.shape[1] // batch_size 
        i = 0  
        for i in range(n_minibatches + 1): 
            X_mini = X[:, i * batch_size:(i + 1)*batch_size] 
            Y_mini = y[:, i * batch_size:(i + 1)*batch_size]
            mini_batches.append((X_mini, Y_mini)) 
        if(X.shape[1] % batch_size == 0 ):
            mini_batches.pop()
        return mini_batches 
    
    def fit(self, X, label, batch_size,num_epochs,optimizer,loss_fn):
        loss_history = []
        itr=0
        plt.ion()
        mini_batches = self.create_mini_batches(X, label, batch_size)
        for epoch in range (num_epochs):
            
            epoch_loss = 0
            for X, label in mini_batches:
                A=X
                for Layer in self.layers: 
                    A=Layer.forward(A)
                batch_loss = loss_fn.forward(A,label)
                epoch_loss += batch_loss
                grad = loss_fn.backward()
                for Layer in self.layers[::-1]:
                    grad = Layer.backward(grad) 
                optimizer.step(self.layers)
            
            loss_history+=[epoch_loss]
            print("Loss at epoch = {} and iteration = {}: {:.2f}".format(epoch,itr,loss_history[-1]))
            itr+=1
            visualization(loss_history)
            plt.show(block=True)
        return loss_history 
        
    def predict(self,X):
        for Layer in self.layers: 
            X = Layer.forward(X)
        return X 
    
'''  
m = model()
m.add(2,2,"Relu")
m.add(2,2,"Relu")
m.Initialize_Parameters()
'''