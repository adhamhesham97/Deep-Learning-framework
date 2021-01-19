from .Activation import activation
import numpy as np

class conv_layer():
    def __init__(self,  Filter=2, num_of_filters=5, Stride=1, padding=0, activation_type="Linear", input_dimensions=(1,1,1)):
        
        self.act_func = activation(activation_type)
        self.S = Stride
        self.P = padding
        
        try:
            FH, FW = Filter    # Filter is tuple (3,4)
        except:
            FH = Filter   # Filter is an integer [square filter]
            FW = Filter 
        
        # input_dimensions is tuple (32, 32, 3)
        n_H_prev, n_W_prev, n_C_prev = input_dimensions 
        
        # layer parameters
        # filters dimensions are (h,w,c,n)
        self.filters = np.random.randn(FH, FW, n_C_prev, num_of_filters) * 0.01
        self.b = np.zeros((1,1,1,num_of_filters))
        
        # the dimensions of output
        self.n_H = int(((n_H_prev+2*self.P-FH)/self.S)+1)
        self.n_W = int(((n_W_prev+2*self.P-FW)/self.S)+1)
        self.n_C = num_of_filters
        
    def zero_pad(self, X, P):
        return np.pad(X, ((0,0), (P,P),(P,P), (0,0)), mode='constant', constant_values = (0,0))
        
    def conv(self, X_slice, filters, b):
        m, FH, FW, n_C_prev = X_slice.shape
        X_slice = X_slice.reshape(m, FH,FW,n_C_prev,1)
        s = np.multiply(X_slice, filters)
        Z = np.sum(s,axis=(1,2,3),keepdims=True)
        Z+=b 
        n_C = b.shape[3]
        Z=Z.reshape(m,n_C)
        return Z
    
    def forward(self, X):
        # X dimensions are (m,h,w,c)
        X_pad = self.zero_pad(X, self.P)
        
        # number of examples
        m, _,_,_ = X.shape
        
        # filter dimensions
        FH, FW, n_C_prev,_ = self.filters.shape
        
        # output array
        Z = np.zeros((m, self.n_H, self.n_W, self.n_C))
        
        for h in range(self.n_H):         
            for w in range(self.n_W):  
                vert_start = h * self.S
                vert_end = vert_start + FH
                horiz_start = w * self.S
                horiz_end = horiz_start + FW
            
                X_slice = X_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]
                Z[:, h, w] = self.conv(X_slice, self.filters, self.b)
                        
        A = self.act_func.forward(Z)
        assert(A.shape == (m, self.n_H, self.n_W, self.n_C))
        self.X = X # cache input for backprop
        return A

    def backward(self, dA):
        # number of examples
        m, _,_,_ = dA.shape
        
        # output arrays
        dX = np.zeros(self.X.shape)                           
        dfilters = np.zeros(self.filters.shape)
        db = np.zeros(self.b.shape)
        
        X_pad = self.zero_pad(self.X, self.P)
        dX_pad = self.zero_pad(dX, self.P)
        
        # filter dimensions
        FH, FW, _,_ = self.filters.shape
        
        # back propagate from activation function
        dA = self.act_func.backward(dA) 
        
        # back propagate from convolution
        for h in range(self.n_H):                   # loop over vertical axis of the output volume
            for w in range(self.n_W):               # loop over horizontal axis of the output volume
                vert_start = h*self.S
                vert_end = vert_start + FH
                horiz_start = w*self.S
                horiz_end = horiz_start + FW
                
                x_slice = X_pad[: ,vert_start:vert_end, horiz_start:horiz_end, :]
                dX_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.sum(self.filters[:,:,:,:] * dA[:, h, w, :].reshape(m,1,1,1,self.n_C), axis=4)
                dfilters += np.sum(x_slice * dA[:, h, w, :].T.reshape(self.n_C,m,1,1,1), axis=1).transpose(1,2,3,0)
                db += np.sum(dA[:, h, w, :],axis=0)
        
        pad=self.P
        dX = dX_pad[:, pad:-pad, pad:-pad, :]
    
        self.dfilters=dfilters
        self.db=db
        
        return dX

    def output_dims(self):
        return self.n_H, self.n_W, self.n_C
    
    def getParams(self):
        return self.filters, self.b

    def getGrads(self):
        return self.dfilters, self.db    

    def setParams(self, filters, b):
        self.filters = filters
        self.b = b
        
    def getLayerParams(self):
        LayerParams = self.filters, self.b, self.S, self.P, self.n_H, self.n_W, self.n_C, self.act_func.activation_type
        return "conv_layer", LayerParams

    def setLayerParams(self, LayerParams):
        self.filters, self.b, self.S, self.P, self.n_H, self.n_W, self.n_C, activation_type = LayerParams
        self.act_func = activation(activation_type)


# forward and backward propagation
'''
activation_type="Linear"
Filter = 2
num_of_filters = 8
input_dimensions = (4, 4, 3)
Stride = 2 
padding = 2
conv = conv_layer(Filter, num_of_filters, Stride, padding, activation_type, input_dimensions)

np.random.seed(1)
X = np.random.randn(10,4,4,3)
filters = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
conv.setParams(filters, b)

A = conv.forward(X)
print("A's mean =", np.mean(A))
print("A[3,2,1] =", A[3,2,1])

dX=conv.backward(A)
dfilters, db = conv.getGrads()
print("dX_mean =", np.mean(dX))
print("dfilters_mean =", np.mean(dfilters))
print("db_mean =", np.mean(db))
'''

# storing and loading

'''
par = conv.getLayerParams()

conv2=conv_layer()
conv2.setLayerParams(par)
A = conv2.forward(X)
print("\n\n\nA's mean =", np.mean(A))
print("A[3,2,1] =", A[3,2,1])

dX=conv2.backward(A)
dfilters, db = conv2.getGrads()
print("dX_mean =", np.mean(dX))
print("dfilters_mean =", np.mean(dfilters))
print("db_mean =", np.mean(db))
'''
