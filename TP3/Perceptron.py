import math
import matplotlib.pyplot as plt
import numpy as np



def step_gen(args):
    def activation(x):
        return np.where(x > 0, 1, -1)

    return activation


activations_gens = { 'step': step_gen}


class Perceptron:


    def __init__(self, x_lenght=2,  activation='step', seed=1, args={}):
        print("perceptron")
        self.structure = x_lenght + 1 #add bias
        self.activation = activations_gens.get(activation)(args)

        if seed != 0:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        self.randomize()

    def randomize(self):
        self.weights = self.rng.uniform(-1, 1, self.structure) #create a vector of weights --> initialize random 1 to -1
    
    def feedforward(self, input):
        x = np.append(input,1)
        H = np.dot(self.weights,x) 
        O = self.activation(H)
        return H,O

    def backpropagation(self, input, expected, O, learning_rate):
        X = np.append(input,1)
        delta = [x*learning_rate*(expected - O) for x in X]
        return delta

    def error(self, dataset,O):
        error = 0
        for index, data in enumerate(dataset):
            input, e = data
            error+=(abs((e-O[index]))/2)
        error = error/len(dataset)
        return error
    


    def train(self, dataset,  target_error=0, epochs=math.inf, learning_rate=0.1):
        errors = []
        
        error = math.inf
        while error > target_error and epochs > 0:
            epochs -= 1
            np.random.shuffle(dataset)
            outputs = []
            for data in dataset:
            
                deltas = []
                input, expected = data
                H, O = self.feedforward(input)
                _deltas = self.backpropagation(input, expected, O, learning_rate)
                if deltas:
                    deltas += _deltas
                else:
                    deltas = _deltas
                outputs.append(O)
                for i in range(len(self.weights)):
                    self.weights[i] += deltas[i]

            error = self.error(dataset,outputs)
            errors.append(error)
        return errors, self.weights



if __name__ == '__main__':
    print("hola soy un perceptron")
