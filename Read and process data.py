# -*- coding: utf-8 -*-


import numpy as np
import os
import pickle

def ReadFile(path):
    path=path.replace("\\","/")
    path=path.replace('"','') 
    filenames = os.listdir(path)
    Features_Train = []
    Label_Train=[]
    if filenames[0][-4:]==".csv":               #MNIST dataset
        
        with open(path+"/test.csv", 'r', encoding='utf-8-sig') as f: 
            FullMatrix_Train = np.genfromtxt(f, dtype=float, delimiter=',')
        with open(path+"/test.csv", 'r', encoding='utf-8-sig') as f: 
                FullMatrix_Test = np.genfromtxt(f, dtype=float, delimiter=',')
    
        Label_Train= FullMatrix_Train[:,0]  
        Temp= FullMatrix_Train[:,1:]
        Features_Train=Temp.transpose()
    
        Label_Test= FullMatrix_Test[:,0]  
        Temp1= FullMatrix_Test[:,1:]
        Features_Test=Temp1.transpose()
      
    else :                                      #CIFAR dataset
        for file in filenames:
            if '_batch' in file:
                with open(path+'/'+file, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    if'test' in file:
                        Label_Test=dict[b'labels']
                        Features_Test=dict[b'data']
                    else:
                       Label_Train.append(dict[b'labels'])
                       Features_Train.append(dict[b'data'])
                    
        Label_Train=np.array(Label_Train)
        Features_Train=np.array(Features_Train)
        Label_Test=np.array(Label_Test)
        Features_Test=np.array(Features_Test)  
        Label_Train=Label_Train.reshape((-1,1))
        Features_Train=Features_Train.reshape((-1,3072))
        Features_Train=Features_Train.transpose()
        Features_Test=Features_Test.transpose()
        
   
    return Label_Train,Features_Train,Label_Test,Features_Test



path=input("Write Directory Path:" )                #Path of dataset



Label_Train,Features_Train,Label_Test,Features_Test=ReadFile(path)
# print(Label_Test)
# print(Features_Test)
# print(Label_Train)
# print(Features_Train)


