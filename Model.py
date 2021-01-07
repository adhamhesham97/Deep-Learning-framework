# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 23:47:45 2021

@author: Aya El Ashry
"""

import numpy as np
import matplotlib.pyplot as plt
from layer import layer
from Optimization import optimization
from Visualization import visualization
from Loss_Functions import Loss_Functions

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
        plt.ion()
        for epoch in range (num_epochs):
            A=X
            for layer in self.layers: 
                A=layer.forward(A)
            loss = Loss_Functions.forward(A,label)
            grad = Loss_Functions.backward()
            for layer in self.layers[::-1]:
                grad = layer.backward(grad) 
            loss_history+=[loss]
            visualization(loss_history)
            print("Loss at epoch = {} and iteration = {}: {}".format(epoch,itr,loss_history[-1]))
            itr+=1
            optimizer.step(self.layers)
        plt.show(block=True)
        return loss_history 
        
    def predict(self,X):
        for layer in self.layers: 
            Prediction = layer.forward(X)
        return Prediction 