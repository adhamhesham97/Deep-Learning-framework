import matplotlib.pyplot as plt
import numpy as np

#this line already exists in fit method of Model class
#loss_history=[]
#add this line to fit method of Model class
#plt.ion()
#input is updated loss_history after each loss calculation in fit method of Model class
def visualization(loss_history):
    x=range(len(loss_history))
    X=np.array(x)+1
    plt.gca().cla()  # optionally clear axes
    plt.plot(X, loss_history, 'o-')
    plt.title("losses from epoch 0 to epoch" + str(i + 1) + "")
    # plt.draw()
    plt.pause(1)
    
#test
'''for i in range(50):
    loss_history+=[i**2-5-i]
    visualization(loss_history)'''

#add this line to fit method of Model class
#plt.show(block=True)#to block closing the figure "optional", maintains reopenning the graph if closed
