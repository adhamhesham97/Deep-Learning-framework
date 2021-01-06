# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 23:47:45 2021

@author: Aya El Ashry
"""

import numpy as np
from layer import layer
from optimization import optimization
from visualization import visualization

class Model:
    
    def __init__(self):
        self.layers=[]
        self.parameters = []

        
    def add(self, in_nodes, out_nodes, activation_type):
        l=layer(in_nodes, out_nodes, activation_type)
        self.layers.append(l)
        self
        
    def Initialize_Parameters(self):        
        for layer in self.layers:
            w,b=layer.getParams()
            w= np.random.randn(w.shape()) * 0.01
            b = np.zeros(b.shape())
            layer.setParams(w, b)

    def fit(self, X, label, batch_size,num_epochs,optimizer,loss_fn):
        loss_history = []
        itr=0
        for epoch in range (num_epochs):
            A=X
            for layer in self.layers: 
                A=layer.forward(A)
            loss = loss.forward(A,label)
            grad = loss.backward()
            for layer in self.layers[::-1]:
                grad = layer.backward(grad) 
            loss_history+=[loss]
            visualization(loss)
            print("Loss at epoch = {} and iteration = {}: {}".format(epoch,itr,loss_history[-1]))
            itr+=1
            optimizer.step(self.layers)
        return loss_history 
        
    def predict(self,X):
        for layer in self.layers: 
            Prediction = layer.forward(X)
        return Prediction 