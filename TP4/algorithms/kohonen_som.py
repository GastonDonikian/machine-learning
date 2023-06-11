import math
import random
import sys

import numpy as np
from numpy import floor


def _calculate_distance(point_1, point_2):
    return np.sqrt(np.sum((point_1 - point_2) ** 2))


def _find_bmu(example, weights):
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
    return w_row, w_col,min_distance


def _initialize_weight_matrix(dimension=2, rows=5, cols=5):
    return np.random.random((cols, rows, dimension))
    # return [[[random.random() for _ in range(dimension)] for _ in range(rows)] for _ in range(cols)]


def _calculate_winning_neuron_update(example, weight, learning_rate):
    # TODO
    return weight + learning_rate * (example - weight)


def _calculate_losing_neuron_update(example, weight, learning_rate, radius):
    if radius == 0:
        return weight
    distance = _calculate_distance(example, weight)
    v_factor = math.exp((-2 * distance) / radius)
    return learning_rate * v_factor * distance * (example - weight)


def kohonen_som(training_set, epochs=2000, eta=0.5, vicinity_radius=3):
    if training_set is None or len(training_set) == 0:
        raise ValueError("Training set can't be empty!")
    dimension = len(training_set[0])
    weight_matrix = _initialize_weight_matrix(dimension=dimension)
    rows, cols, n_array = weight_matrix.shape
    mean_distances_per_epoch = []
    learning_rate = eta
    for epoch in range(epochs):
        print(str((epoch/epochs)*100) + "%", end="\r")
        # EXAMPLE SHOULD BE A NP ARRAY!!!
        min_distaces = []
        for example in training_set:
            # Saco al ganador
            w_row, w_col, min_distance = _find_bmu(example=example, weights=weight_matrix)
            min_distaces.append(min_distance)
            # Actualizo neurona ganadora
            weight_matrix[w_row][w_col] = _calculate_winning_neuron_update(
                example=example,
                weight=weight_matrix[w_row][w_col],
                learning_rate=learning_rate)
            # Actualizo neuronas perdedoras
            for i in range(max(0, w_row - floor(vicinity_radius)), min(rows, w_row + floor(vicinity_radius) + 1)):
                for j in range(max(0, w_col - floor(vicinity_radius)), min(cols, w_col + floor(vicinity_radius) + 1)):
                    if i == w_row and j == w_col:
                        pass
                    weight_matrix[i][j] = _calculate_losing_neuron_update(
                        example=example,
                        weight=weight_matrix[i][j],
                        learning_rate=learning_rate,
                        radius=_calculate_distance(np.array((i, j)), np.array((w_col, w_row))))
        learning_rate = eta * (1 - epoch / epochs)
        mean_distances_per_epoch.append(sum(min_distaces)/len(min_distaces))
        # vicinity_radius = (epochs - epoch) * vicinity_radius / epochs
    return weight_matrix,mean_distances_per_epoch


def predict(example, trained_matrix):
    return _find_bmu(example=example, weights=trained_matrix)


def main():
    trained_matrix = kohonen_som()
    pass


if __name__ == "__main__":
    main()
