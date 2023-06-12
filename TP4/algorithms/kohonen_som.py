import math
import random
import sys

import numpy as np
from numpy import floor



def _calculate_distance(point_1, point_2):
    return np.linalg.norm(np.subtract(point_1, point_2), 2)


def _find_bmu(example, weights,popularity_matrix):
    min_distance = np.inf
    bmu_index = None

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            # por ahora esta verga,
            distance = _calculate_distance(example, weights[i][j])
            if distance < min_distance:
                min_distance = distance
                bmu_index = (i, j)
    w_row, w_col = bmu_index
    popularity_matrix[w_row][w_col] += 1
    return w_row, w_col,min_distance

random_weights = False

def _initialize_weight_matrix(training_set,dimension=2, rows=7, cols=7):
    #weights = [[None for l in range(rows)] for k in range(cols)]
    weights = np.zeros((rows,cols,dimension))
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(0, len(training_set))
            weights[row][col] = training_set[idx].copy()
    return weights
    


def _calculate_winning_neuron_update(example, weight, learning_rate):
    return weight + learning_rate * (example - weight)


def _calculate_losing_neuron_update(example, weight, learning_rate, radius):
    if radius == 0:
        return weight
    distance = _calculate_distance(example, weight)
    v_factor = math.exp((-2 * distance) / radius)
    return weight + learning_rate * v_factor * (example - weight)


def kohonen_som(training_set, epochs=2000, eta=0.5, vicinity_radius=3, rows=7, cols=7):
    if training_set is None or len(training_set) == 0:
        raise ValueError("Training set can't be empty!")
    dimension = len(training_set[0])
    weight_matrix = _initialize_weight_matrix(training_set,dimension=dimension,rows=rows,cols=cols)
    rows, cols, n_array = weight_matrix.shape
    print("rows")
    print(rows)
    print("cols")
    print(cols)
    vicinity_radius = rows
    mean_distances_per_epoch = []
    learning_rate = eta
    radius=vicinity_radius
    popularity_matrix = np.zeros((rows,cols))
    for epoch in range(epochs):
        print(str((epoch/epochs)*100) + "%", end="\r")
        # EXAMPLE SHOULD BE A NP ARRAY!!!
        min_distaces = []
        for example in training_set:
            # Saco al ganador
            w_row, w_col, min_distance = _find_bmu(example=example, weights=weight_matrix, popularity_matrix=popularity_matrix)
            min_distaces.append(min_distance)
            # Actualizo neurona ganadora
            weight_matrix[w_row][w_col] = _calculate_winning_neuron_update(
                example=example,
                weight=weight_matrix[w_row][w_col],
                learning_rate=learning_rate)
            # Actualizo neuronas perdedoras
            for row in range(rows):
                for col in range(cols):
                    if (row!= w_row or col!=w_col) and _calculate_distance([row, col], [w_row, w_col]) <= radius:
                        weight_matrix[row][col] = _calculate_losing_neuron_update(
                        example=example,
                        weight=weight_matrix[row][col],
                        learning_rate=learning_rate,
                        radius=radius)
            
        learning_rate = eta * (1 - epoch / epochs)
        mean_distances_per_epoch.append(sum(min_distaces)/len(min_distaces))
        radius=(epochs-epoch)*vicinity_radius/epochs
        
    return weight_matrix,mean_distances_per_epoch,popularity_matrix


def predict(example, trained_matrix, popularity_matrix):
    return _find_bmu(example=example, weights=trained_matrix, popularity_matrix=popularity_matrix)


def main():
    trained_matrix = kohonen_som()
    pass


if __name__ == "__main__":
    main()
