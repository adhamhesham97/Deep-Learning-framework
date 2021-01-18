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
        A = np.zeros([m, self.n_H, self.n_W, self.n_C])
        
        for i in range(m):                      # loop over the batch of training examples
            for h in range(self.n_H):           # loop over vertical axis of the output volume
                for w in range(self.n_W):       # loop over horizontal axis of the output volume
                    for c in range(self.n_C):   # loop over channels (= #filters) of the output volume
                   
                        # Find the corners of the current "slice"
                        vert_start = h * self.S
                        vert_end = vert_start + self.FH
                        horiz_start = w * self.S
                        horiz_end = horiz_start + self.FW
                        
                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = X[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        
                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                            
                        elif self.mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)    
        
        self.X = X # cache input for backprop
        return A
    
    def backward(self, dA):
        # number of examples
        m, _,_,_ = dA.shape
        
        dX = np.zeros(self.X.shape)
        
        for i in range(m):                       # loop over the training examples                                        
            x = self.X[i]                        # select training example from A_prev 
            for h in range(self.n_H):                 # loop on the vertical axis
                for w in range(self.n_W):             # loop on the horizontal axis
                    for c in range(self.n_C):         # loop over the channels (depth)
                        # Find the corners of the current "slice" 
                        vert_start = h * self.S
                        vert_end = vert_start + self.FH
                        horiz_start = w * self.S
                        horiz_end = horiz_start + self.FW
                        
                        # Compute the backward propagation in both modes.
                        if self.mode == "max":
                            # Use the corners and "c" to define the current slice from a_prev 
                            x_slice = x[vert_start:vert_end, horiz_start:horiz_end, c]
                            # Create the mask from a_prev_slice 
                            mask = (x_slice == np.max(x_slice))
                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) 
                            dX[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        elif self.mode == "average":
                            # Get the value a from dA 
                            da = dA[i,h,w,c]
                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. 
                            dX[i, vert_start: vert_end, horiz_start: horiz_end, c] += da/(self.FW*self.FH)
                            
        return dX
   
    def output_dims(self):
        return self.n_H, self.n_W, self.n_C
    
    def getLayerParams(self):
        return self.S, self.n_H, self.n_W, self.n_C, self.FH, self.FW, self.mode

    def setLayerParams(self, LayerParams):
        self.S, self.n_H, self.n_W, self.n_C, self.FH, self.FW, self.mode = LayerParams


# forward propagation
'''
F = 3
S = 2
input_dims = (4, 4, 3)
maxpool = pool_layer(F, input_dims, S)
avgpool = pool_layer(F, input_dims, S, 'average')

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
maxpool = pool_layer(F, input_dims, S)
avgpool = pool_layer(F, input_dims, S, 'average')

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

