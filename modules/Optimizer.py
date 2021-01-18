import numpy as np
class optimizer:
    
     
    def __init__(self,learningRate=0.1,beta=0):
        self.lr=learningRate
     
   
    def step(self,layers):
        beta=0
        v={}
            
        for layer in layers:
            
            w,b=layer.getParams()
            dw,db=layer.getGrads()
            
            v["dW"+str(layer)]=np.zeros(dw.shape)
            v["db"+str(layer)]=np.zeros(db.shape)
            
            v["dW"+str(layer)]=beta*v["dW"+str(layer)]+(1-beta)*dw
            v["db"+str(layer)]=beta*v["db"+str(layer)]+(1-beta)*db
            
            #updating the layer parameters 
            w=w-self.lr*v["dW"+str(layer)]
            b=b-self.lr*v["db"+str(layer)]
            
            
            #storing the new parameters in layer
            layer.setParams(w,b)
