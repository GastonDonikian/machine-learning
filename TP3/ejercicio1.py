import generate_linearly_separable as ls
import Perceptron as p
import numpy as np
import matplotlib.pyplot as plt
import Svm as s

if __name__ == '__main__':
    ##EJ 1.1
    # category_one, category_minus_one = ls.generate_points_linearly_separable(f=lambda x: x)
    # dataset = []
    # dataset += [[x, 1] for x in category_one]
    # dataset += [[x, -1] for x in category_minus_one]
    # perceptron = p.Perceptron(2, activation='step', seed=1)
    # error, weights = perceptron.train(dataset, learning_rate=1, epochs=1000)
    # plt.scatter(*zip(*category_one), color='red')
    # plt.scatter(*zip(*category_minus_one), color='blue')
    # x = np.linspace(0,5,2)
    # y = (-weights[2] -weights[0]*x )/weights[1]
    # plt.plot(x, y, '-g')
    # plt.grid()
    # plt.show()

    ##EJ 1.3
    # category_one, category_minus_one = ls.generate_points_linearly_separable(wrong=True, f=lambda x: x)
    # dataset = []
    # dataset += [[x, 1] for x in category_one]
    # dataset += [[x, -1] for x in category_minus_one]
    # perceptron = p.Perceptron(2, activation='step', seed=1)
    # error, weights = perceptron.train(dataset, learning_rate=1, epochs=100)
    # plt.scatter(*zip(*category_one), color='red')
    # plt.scatter(*zip(*category_minus_one), color='blue')
    # x = np.linspace(0,5,2)
    # y = (-weights[2] -weights[0]*x )/weights[1]
    # plt.plot(x, y, '-g')
    # plt.grid()
    # plt.show()

    ##EJ 1.4
    category_one, category_minus_one = ls.generate_points_linearly_separable(f=lambda x: x)
    #category_one, category_minus_one = ls.generate_points_linearly_separable(wrong=True, f=lambda x: x)
    plt.scatter(*zip(*category_one), color='red')
    plt.scatter(*zip(*category_minus_one), color='blue')
    
    dataset = []
    dataset += [[x, 1] for x in category_one]
    dataset += [[x, -1] for x in category_minus_one]
    svm = s.SVM()
    weights, b = svm.svg_one_sample(dataset)
    x = np.linspace(0,5,2)
    y = (-b -weights[0]*x )/weights[1]
    plt.plot(x, y, '-g')
    plt.grid()
    plt.show()