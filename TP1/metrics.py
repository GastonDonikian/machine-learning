import numpy as np
import pandas as pd
import copy
import random

iterations = 10
SEED = 2023
random.seed(SEED)

def cross_validation(dataset,  k):
    n = int(len(dataset)/k)
    dataset.sample(frac=1) #ver si mod o si devuelve
    
    test = dataset.tail(-n)
    
    return dataset, test
    
    


def confusion_matrix(classes, predicted, expected):
    matrix = np.zeros((len(classes)+1, len(classes)+1))
    for i,j  in expected, predicted:
        matrix[get_index(i)][get_index(j)] += 1
    print(matrix)
    return matrix

def get_index(category):
    if category == 'deportes':
        return 0
    if category == 'destacadas':
        return 1
    if category == 'nacional':
        return 2
    if category == 'salud':
        return 3
    


def accuracy(network, dataset, epsilon=0.1):
    good = 0
    bad = 0
    for i in range(len(dataset)):
        expected = dataset[i][1][0]
        o = network.feedforward(dataset[i][0])
        if (np.abs(expected-o) <= epsilon):
            good += 1
        else:
            bad += 1
    p = good/(good + bad)
    return p