import numpy as np
import os
import pickle




def ReadFile(path):
    print("Loading data...")
    path=path.replace("\\","/")
    path=path.replace('"','') 
    filenames = os.listdir(path)
    Features_Train = []
    Label_Train=[]
    if filenames[0][-4:]==".csv":               #MNIST dataset
        
        with open(path+"/train.csv", 'r', encoding='utf-8-sig') as f: 
            FullMatrix_Train = np.genfromtxt(f, dtype=float, delimiter=',')
        with open(path+"/test.csv", 'r', encoding='utf-8-sig') as f: 
                FullMatrix_Test = np.genfromtxt(f, dtype=float, delimiter=',')
                
        #  Shuffling data  
        np.random.shuffle(FullMatrix_Train)
        np.random.shuffle(FullMatrix_Test)
    
        Label_Train= FullMatrix_Train[:,0]  
        Temp= FullMatrix_Train[:,1:]
        Features_Train=Temp.transpose()
        
       
    
        Label_Test= FullMatrix_Test[:,0]
        Temp1= FullMatrix_Test[:,1:]
        Features_Test=Temp1.transpose()
        
        Features_Train = Features_Train.T.reshape(-1,28,28,1).transpose(0,3,1,2)
        Features_Test = Features_Test.T.reshape(-1,28,28,1).transpose(0,3,1,2)
        
        
        
        
      
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
        Features_Train = Features_Train.T.reshape(-1,3,32,32)
        Features_Test = Features_Test.T.reshape(-1,3,32,32)



    # Normalization
    Features_Train=Features_Train/255
    Features_Test=Features_Test/255

    
    
    return Label_Train,Features_Train,Label_Test,Features_Test







# path=input("Write Directory Path:" )                #Path of dataset
# Label_Train,Features_Train,Label_Test,Features_Test=ReadFile(path)
# print(Label_Test)
# print(Features_Test)
# print(Label_Train)
# print(Features_Train)






# visualize mnist
'''
Label_Train,Features_Train,Label_Test,Features_Test=ReadFile("H:\\4th comp\\NN\\MNISTcsv")
import matplotlib.pyplot as plt
i=20 # image number
pixels = Features_Test[:,i].reshape((28, 28))
plt.title('Label is {}'.format(Label_Test[i]))
plt.imshow(pixels, cmap='gray')
'''

# visualize cifar-10
'''
import matplotlib.pyplot as plt
Label_Train,Features_Train,Label_Test,Features_Test=ReadFile("H:\\4th comp\\NN\\cifar-10-batches-py")
i=9 # image number
im=Features_Test[:,i]
im_r = im[0:1024].reshape(32, 32)
im_g = im[1024:2048].reshape(32, 32)
im_b = im[2048:].reshape(32, 32)
img = np.dstack((im_r, im_g, im_b))
plt.title('Label is {}'.format(Label_Test[i]))
plt.imshow(img) 
plt.show()
'''
