import generate_linearly_separable as ls
import Perceptron as p
import numpy as np
import matplotlib.pyplot as plt
import Svm as s
import random
import metrics as m
import numpy as np

def distance_point_to_line(point, line):
    p = np.array(point)
    a, b, c = line
    return abs(a * p[0] + b* p[1] + c) / np.sqrt(a**2 + b**2)

def find_nearest_points(points, line):
    distances = [distance_point_to_line(point, line) for point in points]
    sorted_indices = np.argsort(distances)
    nearest_points = [points[idx] for idx in sorted_indices[:3]]
    return nearest_points

def optimal_hyperplane(category_one, category_minus_one,weights,seed=100):
    random.seed(seed)
    #random_choice_o = random.choices(category_one, k=2)
    #random_choice_m = random.choices(category_minus_one, k=1)
    near_point_one = find_nearest_points(category_one,weights)
    near_point_minus = find_nearest_points(category_minus_one,weights)
    random_choice_o = random.choices(near_point_one, k=2)
    random_choice_m = random.choices(near_point_minus, k=1)
    #agarro dos puntos de la clase 1
    x1, y1 = random_choice_o[0]
    x2, y2 = random_choice_o[1]

    #agarro uno de la clase -1
    xb, yb = random_choice_m[0]

    #pendiente y b recta de los puntos de la clase 1
    m = (y2 - y1) / (x2 - x1)

    #PUNTOMEDIO DE los puntos
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2

    #PUNTOMEDIO entre el punto de la clase -1 y el punto medio entre los puntos de la clase 1
    xm_mid = (xm + xb) / 2
    ym_mid = (ym + yb) / 2

    b_mid = ym_mid - m*xm_mid

    return m,b_mid

def ej1y2():
    category_one, category_minus_one = ls.generate_points_linearly_separable(seed=10,f=lambda x: x)

    dataset = []
    dataset += [[x, 1] for x in category_one]
    dataset += [[x, -1] for x in category_minus_one]
    perceptron = p.Perceptron(2, activation='step', seed=1)
    error, weights = perceptron.train(dataset, learning_rate=1, epochs=1000)
    plt.scatter(*zip(*category_one), color='red')
    plt.scatter(*zip(*category_minus_one), color='blue')
    plt.xlabel("X")
    plt.ylabel("Y")

    x = np.linspace(0,5,2)
    y = (-weights[2] -weights[0]*x )/weights[1]
    m,b = optimal_hyperplane(category_one,category_minus_one,weights,40)
    plt.plot(x, y, '-g',label='perceptron')
    y_optimal=(m*x + b)
    plt.plot(x, y_optimal, '-m', label='near_optimal')
    plt.legend()
    plt.xlim(-0.05, 5.05)  # Set the x-axis limits from 0 to 6
    plt.ylim(-0.05, 5.05)  # Set the y-axis limits from 0 to 12
    plt.grid()
    plt.show()

def ej3():
     ##EJ 1.3
    category_one, category_minus_one = ls.generate_points_linearly_separable(wrong=True, f=lambda x: x)
    dataset = []
    dataset += [[x, 1] for x in category_one]
    dataset += [[x, -1] for x in category_minus_one]
    perceptron = p.Perceptron(2, activation='step', seed=1)
    error, weights = perceptron.train(dataset, learning_rate=1, epochs=100)
    plt.scatter(*zip(*category_one), color='red')
    plt.scatter(*zip(*category_minus_one), color='blue')
    x = np.linspace(0,5,2)
    y = (-weights[2] -weights[0]*x )/weights[1]
    plt.plot(x, y, '-g')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()

def ej4():
    category_one, category_minus_one = ls.generate_points_linearly_separable(f=lambda x: x)
    category_one, category_minus_one = ls.generate_points_linearly_separable( f=lambda x: x)
    dataset = []
    dataset += [[x, 1] for x in category_one]
    dataset += [[x, -1] for x in category_minus_one]
    svm = s.SVM()
    weights, b = svm.svg_one_sample(dataset,2)
    x = np.linspace(0,5,2)
    y = (-b -weights[0]*x )/weights[1]
    plt.plot(x, y, '-g')
    plt.scatter(*zip(*category_one), color='red')
    plt.scatter(*zip(*category_minus_one), color='blue')    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()

    # dataset = m.cross_validation(dataset,6)
    # for i in range(6):
    #     test, training = m.choose_test(i,dataset)
    #     svm = s.SVM()
    #     weights, b = svm.svg_one_sample(training,2)
    #     x = np.linspace(0,5,2)
    #     y = (-b -weights[0]*x )/weights[1]
    #     plt.plot(x, y, '-g')
    #     plt.scatter(*zip(*category_one), color='red')
    #     plt.scatter(*zip(*category_minus_one), color='blue')    
    #     plt.grid()
    #     plt.show()
    #     print(s.compute_cost(weights,b,test))


    

if __name__ == '__main__':
    #ej1y2()
    #ej3()
    ej4()
  
