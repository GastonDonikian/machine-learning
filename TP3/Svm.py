import generate_linearly_separable as ls
import Perceptron as p
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class SVM: 
    def __init__(self, c= 1, kw=0.001, kb=0.05, seed=0, learning_rate = 0.00001):
        self.c = c
        self.kw = kw
        self.kb = kb
        self.b = 0
        self.max_epochs = 10000
        self.seed = seed
        self.learning_rate = learning_rate
        if seed != 0:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        

    def svg_one_sample(self, dataset):
        self.weights = self.rng.uniform(-1, 1, 2)
        b = 0
        prev_cost = float("inf")
        initial_rate = self.learning_rate 
        best_cost = prev_cost 
        bestW = self.weights
        bestB = b
        for epoch in range(1, self.max_epochs):
            self.kw -= self.kw / epoch
            self.kb -= self.kb / epoch
            # Get one randon sample (stochastic)
            np.random.shuffle(dataset)
            input, expected = dataset[0]

            w_gradient,b_gradient = compute_gradient(self.weights,b,np.array(input),expected,self.c)
            self.weights = self.weights - (self.learning_rate * w_gradient)
            b = b - (self.learning_rate * b_gradient)
            # Now check the cost function
            cost = compute_cost(self.weights, b, dataset)
            #print( "Epoch is: {} and Cost is: {} and lr {}".format(epoch, cost, self.learning_rate))
            # Store the best result
            if cost < best_cost:
                best_cost = cost
                bestW = self.weights
                bestB = b
                
        return bestW,bestB

def compute_gradient(W,b,x,y,c):
    output_grad = (y * (np.dot(x,W) + b))
    dl = 0
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


