import numpy as np 
class Loss_Functions():
    def __init__(self,loss_type):
        self.loss_type=loss_type
        #self.pred
        #self.labels
        
    def forward(self,pred,labels):
        self.m=labels.shape[1]
        self.pred=pred
        self.labels=labels
        
        if self.loss_type=="LOG":
            return (1/self.m)*np.sum(-np.log(np.absolute((labels/2)-0.5+pred)))
        
        elif self.loss_type=="MEAN":
            return np.sum(np.power(pred-labels,2))/(2*self.m)
        
        elif self.loss_type=="CROSSENTROPY":
            return (-1/self.m)*np.sum(np.multiply(labels,np.log(pred))+np.multiply(1-labels,np.log(1-pred)))

    def backward(self):
        if self.loss_type=="LOG":
            return (((-1/self.pred)*((1+self.labels)/2)) + ((1/(1-self.pred))*((1-self.labels)/2)))
        
        elif self.loss_type=="MEAN":
            return (self.pred-self.labels)
        
        elif self.loss_type=="CROSSENTROPY":
            return ((-self.labels/self.pred)+(1-self.labels/1-self.pred))
        
        
        
#lossfn=Loss_Functions('MEAN') 
#pred=np.array([[1,1,1]])  
#labels=np.array([[1,1,-1]]) 

#loss=lossfn.forward(pred, labels)  
#grad=lossfn.backward()  
#print(loss)
#print(grad)


        
