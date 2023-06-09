import numpy as np


def _calculate_distance(point_1, point_2):
    return np.sqrt(np.sum((point_1 - point_2) ** 2))


def find_bmu(example, weights):
    min_distance = np.inf
    bmu_index = None

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
        # por ahora esta verga,
            distance = _calculate_distance(example, weights[i][j])
            if distance < min_distance:
                min_distance = distance
                bmu_index = (i,j)

    return bmu_index


def initialize_weight_matrix(rows=5, cols=5):
   return np.random.rand(rows, cols)

def calculate_winning_neuron_update(example, weight, learning_rate):
    # TODO
    return weight

def calculate_lossing_neuron_update(example, weight, learning_rate):
    # TODO
    return weight

def kohonoen_som(training_set, epochs=2000, learning_rate=0.5, vicinity_radius=3):
    weight_matrix = initialize_weight_matrix()
    rows, cols = weight_matrix.shape
    for _ in range(epochs):
        # EXAMPLE SHOULD BE A NP ARRAY!!!
        for example in training_set:
            # Saco al ganador
            w_row, w_col = find_bmu(example=example,weights=weight_matrix)
            # Actualizo neurona ganadora
            weight_matrix[w_row][w_col] = calculate_winning_neuron_update(
                                                example=example,
                                                weight=weight_matrix[w_row][w_col],
                                                learning_rate=learning_rate)
            # Actualizo neuronas perdedoras
            for i in range(max(0, w_row - vicinity_radius), min(rows, w_row + vicinity_radius + 1)):
                for j in range(max(0, w_col - vicinity_radius), min(cols, w_col + vicinity_radius + 1)):
                    if i == w_row and j == w_col:
                        pass
                    weight_matrix[i][j] = calculate_lossing_neuron_update(
                                                example=example,
                                                weight=weight_matrix[i][j],
                                                learning_rate=learning_rate)
        # El ppt dice aumentarlo, pero lo reduzco pq tiene mas sentido
        learning_rate *= 0.5
        # supuewstamente hay que aumentar el vicinity radius, no se.





def main():
    pass


if __name__ == "__main__":
    main()
