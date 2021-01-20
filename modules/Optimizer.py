class optimizer:
    
     
    def __init__(self,learningRate=0.1,beta=0):
        self.lr=learningRate
        self.beta=beta
        self.v={}
        
    
    def set_v(self,v):
        self.v=v
        
    def get_v(self):
        return self.v
    
    def step(self,layers):
        
        
        i=0
        for layer in layers:
            
            grads = layer.getGrads()
            
            if(grads is not None):
                
                w,b=layer.getParams()
                dw,db=grads
                
                self.v["dW"+str(i+1)]=self.beta*self.v.get("dW"+str(i+1),0)+(1-self.beta)*dw
                self.v["db"+str(i+1)]=self.beta*self.v.get("db"+str(i+1),0)+(1-self.beta)*db
                
                #updating the layer parameters 
                w=w-self.lr*self.v["dW"+str(i+1)]
                b=b-self.lr*self.v["db"+str(i+1)]
                i=i+1
                
                #storing the new parameters in layer
                layer.setParams(w,b)
                