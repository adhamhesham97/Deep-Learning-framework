# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 23:47:45 2021

@author: Aya El Ashry
"""

import numpy as np
import matplotlib.pyplot as plt
from .Layer import layer
from .Visualization import visualization

class model:
    
    def __init__(self):
        self.layers=[]
        
        
    def add(self, activation_type, in_nodes=0, out_nodes=0, Padding=0, activation_type_conv=''):
        
        if(activation_type=='batch_norm'):
            l=layer(activation_type)
        
        elif(activation_type=='maxpool'):
            l=layer(activation_type, in_nodes, out_nodes)
        
        elif(activation_type=='conv'):
            l=layer(activation_type, in_nodes, out_nodes, Padding)
        
        elif(activation_type=='flatten'):
            l=layer(activation_type)
        
        else:
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
        Lambda=loss_fn.getLambda()
        for epoch in range (num_epochs):
            
            epoch_loss = 0
            for X, label in mini_batches:
                A=X
                weights_sum=0
                for Layer in self.layers: 
                    A, w_squared = Layer.forward(A)
                    weights_sum += w_squared 
                    
                batch_loss = loss_fn.forward(A,label,weights_sum)
                epoch_loss += batch_loss
                grad = loss_fn.backward()
                for Layer in self.layers[::-1]:
                    grad = Layer.backward(grad,Lambda) 
                optimizer.step(self.layers)
            
            loss_history+=[epoch_loss]
            print("Loss at epoch = {} and iteration = {}: {:.2f}".format(epoch,itr,loss_history[-1]))
            itr+=1
            visualization(loss_history)
            plt.show(block=True)
        return loss_history 
        
    def predict(self,X):
        for Layer in self.layers: 
            X, _ = Layer.forward(X)
        return X 
    
    def getParams(self):
        List=[]
        for Layer in self.layers:
            List.append(Layer.getLayerParams())
        return List
    
    def setParams(self, List):
        for LayerParams in List:
            Layer = layer()
            Layer.setLayerParams(LayerParams)
            self.layers.append(Layer)
    
  
'''  
m = model()
m.add(2,2,"Relu")
m.add(2,2,"Relu")
m.Initialize_Parameters()
'''
