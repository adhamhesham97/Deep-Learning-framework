import numpy as np
class optimizer:
    
     
    def __init__(self,learningRate=0.1,beta=0):
        self.lr=learningRate
        self.beta=beta
        
        
        
   
    def step(self,layers):
        
        v={}
        i=0
        for layer in layers:
            
            grads = layer.getGrads()
            
            if(grads is not None):
                dw,db = grads
                w,b=layer.getParams()
                v["dW"+str(i+1)]=np.zeros(dw.shape)
                v["db"+str(i+1)]=np.zeros(db.shape)
               
                v["dW"+str(i+1)]=self.beta*v["dW"+str(i+1)]+(1-self.beta)*dw
                v["db"+str(i+1)]=self.beta*v["db"+str(i+1)]+(1-self.beta)*db
                
                #updating the layer parameters 
                w=w-self.lr*v["dW"+str(i+1)]
                b=b-self.lr*v["db"+str(i+1)]
                i=i+1
                
                #storing the new parameters in layer
                layer.setParams(w,b)
            
