import numpy as np

class pool_layer():
    def __init__(self, Filter=2, Stride=1, mode = "max", input_dimensions=(1,1,1)):
        
        self.mode = mode # maxpool (max) or average pool (average)
        self.S = Stride
        
        try:
            FH, FW = Filter    # Filter is tuple (3,4)
        except:
            FH = Filter   # Filter is an integer [square filter]
            FW = Filter 
       
        self.FH = FH
        self.FW = FW
        
        # input_dimensions is tuple (32, 32, 3)
        n_H_prev, n_W_prev, n_C_prev = input_dimensions 
        
        # the dimensions of output
        self.n_H = int(((n_H_prev-FH)/self.S)+1)
        self.n_W = int(((n_W_prev-FW)/self.S)+1)
        self.n_C = n_C_prev
        
        
    def forward(self, X):    
        # X dimensions are (m,h,w,c)
        # number of examples
        m, _,_,_ = X.shape
        
        # output array
        A = np.zeros((m, self.n_H, self.n_W, self.n_C))
        
        for h in range(self.n_H):           # loop over vertical axis of the output volume
            for w in range(self.n_W):       # loop over horizontal axis of the output volume
                
                    vert_start = h * self.S
                    vert_end = vert_start + self.FH
                    horiz_start = w * self.S
                    horiz_end = horiz_start + self.FW
                    
                    x_slice = X[:, vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if self.mode == "max":
                        A[:, h, w, :] = np.max(x_slice,axis=(1,2))
                        
                    elif self.mode == "average":
                        A[:, h, w, :] = np.mean(x_slice,axis=(1,2))    
        
        self.X = X # cache input for backprop
        return A
    
    def backward(self, dA):
        # number of examples
        m, _,_,_ = dA.shape
        
        # output vector
        dX = np.zeros(self.X.shape)
        
        n_C = self.n_C
        x = self.X
        for h in range(self.n_H):                 # loop on the vertical axis
            for w in range(self.n_W):             # loop on the horizontal axis
                    
                vert_start = h
                vert_end = vert_start + self.FH
                horiz_start = w
                horiz_end = horiz_start + self.FW
                
                if self.mode == "max":
                    x_slice = x[:,vert_start:vert_end, horiz_start:horiz_end, :]
                    mask = (x_slice == np.amax(x_slice,axis=(1,2),keepdims=True))
                    # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) 
                    dX[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.multiply(mask, dA[:, h, w, :].reshape(m,1,1,n_C))
                
                elif self.mode == "average":
                    # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. 
                    dX[:, vert_start: vert_end, horiz_start: horiz_end, :] += dA[:,h,w,:].reshape(m,1,1,n_C)/(self.FW*self.FH)
                        
        return dX
    
    def getGrads(self):
        return None
    
    def output_dims(self):
        return self.n_H, self.n_W, self.n_C
    
    def getLayerParams(self):
        LayerParams = self.S, self.n_H, self.n_W, self.n_C, self.FH, self.FW, self.mode
        return "pool_layer", LayerParams

    def setLayerParams(self, LayerParams):
        self.S, self.n_H, self.n_W, self.n_C, self.FH, self.FW, self.mode = LayerParams


# forward propagation
'''
F = 3
S = 2
input_dims = (4, 4, 3)
maxpool = pool_layer(F, S, "max", input_dims)
avgpool = pool_layer(F, S, 'average', input_dims)

np.random.seed(1)
X = np.random.randn(2, 4, 4, 3)
print("maxpool forward:\n",maxpool.forward(X))
print("avgpool forward:\n",avgpool.forward(X))
'''

# back propagation
'''
F = 2
S = 1
input_dims = (5, 3, 2)
maxpool = pool_layer(F, S, "max", input_dims)
avgpool = pool_layer(F, S, 'average', input_dims)

np.random.seed(1)
X = np.random.randn(5, 5, 3, 2)
A = maxpool.forward(X)
A = avgpool.forward(X)
dA = np.random.randn(5, 4, 2, 2)


dX = maxpool.backward(dA)
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dX[1,1] = ', dX[1,1])  
print()
dX = avgpool.backward(dA)
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dX[1,1] = ', dX[1,1])
'''

# storing and loading

'''
par = maxpool.getLayerParams()

maxpool2 = pool_layer()
maxpool2.setLayerParams(par)
A = maxpool2.forward(X)

dX = maxpool2.backward(dA)
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dX[1,1] = ', dX[1,1])  
print()
'''

