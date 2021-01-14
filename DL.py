from Optimizer import optimizer
from Loss_Function import loss_Function
from Model import model
from metrics import *
from Visualization import * # to be changed
from Read_and_process_data import *


def hot_one(labels, num_classes):
    num_of_examples = labels.shape[0]
    hot_one = np.zeros((num_classes, num_of_examples))
    for i in range(num_of_examples):
        hot_one [int(labels[i])] [i] = 1
    return hot_one