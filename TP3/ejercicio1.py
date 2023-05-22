import generate_linearly_separable as ls
import Perceptron as p
import numpy as np
import matplotlib.pyplot as plt
import Svm as s
import random



	
def optimal_hyperplane(category_one, category_minus_one,seed=200):
    random.seed(seed)
    random_choice_o = random.choices(category_one, k=2)
    random_choice_m = random.choices(category_minus_one, k=1)
    x1, y1 = random_choice_o[0]
    x2, y2 = random_choice_o[1]
    xb, yb = random_choice_m[0]
    #pendiente y b recta de los puntos de la clase a
    m = (y2 - y1) / (x2 - x1)
    #PUNTOMEDIO DE los puntos
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2
    #PUNTOMEDIO entre el punto de la clase b y el punto medio entre los puntos de la clase a
    xm_mid = (xm + xb) / 2
    ym_mid = (ym + yb) / 2

    b_mid = ym_mid - m*xm_mid

    return m,b_mid

if __name__ == '__main__':
    ##EJ 1.1
    category_one, category_minus_one = ls.generate_points_linearly_separable(seed=10,f=lambda x: x)
    m,b = optimal_hyperplane(category_one,category_minus_one,40)
    dataset = []
    dataset += [[x, 1] for x in category_one]
    dataset += [[x, -1] for x in category_minus_one]
    perceptron = p.Perceptron(2, activation='step', seed=1)
    error, weights = perceptron.train(dataset, learning_rate=1, epochs=1000)
    plt.scatter(*zip(*category_one), color='red')
    plt.scatter(*zip(*category_minus_one), color='blue')


    x = np.linspace(0,5,2)
    y = (-weights[2] -weights[0]*x )/weights[1]
    plt.plot(x, y, '-g')
    y_optimal=(m*x + b)
    plt.plot(x, y_optimal, '-m')
    plt.xlim(-0.05, 5.05)  # Set the x-axis limits from 0 to 6
    plt.ylim(-0.05, 5.05)  # Set the y-axis limits from 0 to 12
    plt.grid()
    plt.show()

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
    ##TODO: GRAFICOS DE VALIDACIÓN CRUZADA PARA SACAR EL C OPTIMO, (KW Y KB CONJUNTO)
    #category_one, category_minus_one = ls.generate_points_linearly_separable(f=lambda x: x)
    # category_one, category_minus_one = ls.generate_points_linearly_separable(wrong=True, f=lambda x: x) 
    # plt.scatter(*zip(*category_one), color='red')
    # plt.scatter(*zip(*category_minus_one), color='blue')
    
    # dataset = []
    # dataset += [[x, 1] for x in category_one]
    # dataset += [[x, -1] for x in category_minus_one]
    # svm = s.SVM()
    # weights, b = svm.svg_one_sample(dataset)
    # x = np.linspace(0,5,2)
    # y = (-b -weights[0]*x )/weights[1]
    # plt.plot(x, y, '-g')
    # plt.grid()
    # plt.show()