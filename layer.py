from activation import activation
import numpy as np


class layer():
    def __init__(self, in_nodes, out_nodes, activation_type):
        self.w = np.random.randn(out_nodes, in_nodes)*0.01
        self.dw = np.zeros((out_nodes, in_nodes))
        self.b = np.zeros((out_nodes, 1))
        self.db = np.zeros((out_nodes, 1))
        self.act_func = activation(activation_type)

    def forward(self, X):
        Z = self.w @ X + self.b # Z = w * X + b
        output = self.act_func.forward(Z) # A = activation(Z)
        self.X = X # cache input to use it in back prop
        return output

    def backward(self, dA):
        dZ = self.act_func.backward(dA)  # dZ = dA . g'(Z) (element wise product)
        self.dw = dZ @ self.X.T  # dw = dZ * X.T
        self.db = (1/dZ.shape[0]) * np.sum(dZ, axis=0, keepdims=True) # db = 1/m * sum(dZ)
        grad_input = self.w.T @ dZ
        return grad_input

    def getParams(self):
        return self.w, self.b
    
    def getGrads(self):
        return self.dw, self.db
    
    def setParams(self, w, b):
        self.w = w
        self.b = b

'''
in_nodes=2
out_nodes=3
activation_type = "Relu" #(Relu, Sigmoid, Linear)
L = layer(in_nodes, out_nodes, activation_type)
w,b = L.getParams()
w = w + 5
b = np.ones((out_nodes, 1))
L.setParams(w, b)
print('Weights=\n',w,"\nb=\n",b)

X = np.array([
            [0, 1, 2],       
            [1, 4, 3]
          ])
A = L.forward(X)
print("\n\nfor input=\n",X,"\noutput=\n",A)

dA = np.array([
             [ 1, 9, 7],
             [ 6, 1, 2],
             [ 4, 5, 8]
             ])
dX = L.backward(dA)
print("\n\nfor dA=\n",dA,"\ndX=\n",dX)
'''


