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

    def fit(self, X, label, batch_size,num_epochs,optimizer,loss_fn):
        loss_history = []
        itr=0
        plt.ion()
        for epoch in range (num_epochs):
            A=X
            for Layer in self.layers: 
                A=Layer.forward(A)
            loss = loss_fn.forward(A,label)
            grad = loss_fn.backward()
            for Layer in self.layers[::-1]:
                grad = Layer.backward(grad) 
            loss_history+=[loss]
            visualization(loss_history)
            print("Loss at epoch = {} and iteration = {}: {}".format(epoch,itr,loss_history[-1]))
            itr+=1
            optimizer.step(self.layers)
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