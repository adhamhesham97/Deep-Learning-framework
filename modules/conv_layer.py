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
        
    def conv_single_step(self, X_slice_prev, W, b):
        s = np.multiply(X_slice_prev, W)
        # Sum over all entries of the volume s.
        Z = np.sum(s)
        # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
        Z+=float(b) 
        return Z
    
    def forward(self, X):
        # X dimensions are (m,h,w,c)
        X_pad = self.zero_pad(X, self.P)
        
        # number of examples
        m, _,_,_ = X.shape
        
        # filter dimensions
        FH, FW, _,_ = self.filters.shape
        
        # output array
        Z = np.zeros((m, self.n_H, self.n_W, self.n_C))
        
        for i in range(m):                      # loop over the batch of training examples
            X_prev_pad = X_pad[i,:,:,:]         # Select ith training example's padded activation
            for h in range(self.n_H):           # loop over vertical axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                vert_start = h * self.S
                vert_end = vert_start + FH
                for w in range(self.n_W):       # loop over horizontal axis of the output volume
                    # Find the horizontal start and end of the current "slice" (≈2 lines)
                    horiz_start = w * self.S
                    horiz_end = horiz_start + FW
                    for c in range(self.n_C):   # loop over channels (= #filters) of the output volume
                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        X_slice_prev = X_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                        Z[i, h, w, c] = self.conv_single_step(X_slice_prev, self.filters[:,:,:,c], self.b[:,:,:,c])
                        
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
        for i in range(m):                       # loop over the training examples
        
            # select ith training example from A_prev_pad and dA_prev_pad
            x_pad = X_pad[i]
            dx_pad = dX_pad[i]
            
            for h in range(self.n_H):                   # loop over vertical axis of the output volume
                for w in range(self.n_W):               # loop over horizontal axis of the output volume
                    for c in range(self.n_C):           # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice"
                        vert_start = h*self.S
                        vert_end = vert_start + FH
                        horiz_start = w*self.S
                        horiz_end = horiz_start + FW
                        
                        # Use the corners to define the slice from a_prev_pad
                        x_slice = x_pad[vert_start:vert_end, horiz_start:horiz_end, :]
    
                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        dx_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.filters[:,:,:,c] * dA[i, h, w, c]
                        dfilters[:,:,:,c] += x_slice * dA[i, h, w, c]
                        db[:,:,:,c] += dA[i, h, w, c]
                        
            
            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            pad=self.P
            dX[i, :, :, :] = dx_pad[pad:-pad, pad:-pad, :]
        
        
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
conv = conv_layer(activation_type, Filter, num_of_filters, input_dimensions, Stride, padding)

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
