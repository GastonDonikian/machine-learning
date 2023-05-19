import generate_linearly_separable as ls
import Perceptron as p
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    category_one, category_minus_one = ls.generate_points_linearly_separable(f=lambda x: x)
    dataset = []
    dataset += [[x, 1] for x in category_one]
    dataset += [[x, -1] for x in category_minus_one]
    perceptron = p.Perceptron(2, activation='step', seed=1)
    print(perceptron.train(dataset, learning_rate=1, epochs=1000))

