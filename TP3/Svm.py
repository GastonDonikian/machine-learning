import generate_linearly_separable as ls
import Perceptron as p
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class SVM: 
    def __init__(self, c= 100, seed=0, initial_rate = 0.00001,max_epochs=10000):
        self.c = c
        self.b = 0
        self.max_epochs = max_epochs
        self.seed = seed
        self.initial_rate = initial_rate
        if seed != 0:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def svg_one_sample(self, dataset, structure):
        self.weights = np.random.rand(structure)
        b = 0
        prev_cost = float("inf")
        learning_rate = self.initial_rate
        best_cost = prev_cost 
        bestW = self.weights
        bestB = b
        for epoch in range(1, self.max_epochs):
            # Get one randon sample (stochastic)
            np.random.shuffle(dataset)
            input, expected = dataset[0]
            learning_rate = self.initial_rate*np.exp(-0.0001*epoch)
            
            w_gradient,b_gradient = compute_gradient(self.weights,b,np.array(input),expected,self.c)
            self.weights = self.weights - (learning_rate * w_gradient)
            b = b - (learning_rate * b_gradient)
            # Now check the cost function
            cost = compute_cost(self.weights, b, dataset)
            #print( "Epoch is: {} and Cost is: {} and lr {}".format(epoch, cost, learning_rate))
            # Store the best result
            if cost < best_cost:
                best_cost = cost
                bestW = self.weights
                bestB = b
                
        return bestW,bestB

def compute_gradient(W,b,x,y,c):
    output_grad = (y * (np.dot(x,W) + b))
    di = 0
    bi = 0

    if output_grad >= 1:
        di = W
        bi = 0
    else:
        di = W - (c * y * x)
        bi = (c * y * (-1))
    
    return di,bi
        
        

   
def compute_cost(weights,b,dataset):
    cost = 0
    for data in dataset:
        input, expected = data
        output  = np.dot(weights, input) + b
        cost += 0.5 * (expected-output)**2  # error cuadrático medio (MSE)

    
    # Calcular el costo promedio dividiendo por el número de ejemplos de entrenamiento
    cost /= len(dataset)
    return cost  


def accuracy(weights, b, dataset):
    good = 0
    total = len(dataset)
    for data in dataset:
        input, expected = data
        output  = np.dot(weights, input) + b
        if (output*expected) > 0:
            good+=1
    
    # Calcular el costo promedio dividiendo por el número de ejemplos de entrenamiento
    result = good/ total
    return result  