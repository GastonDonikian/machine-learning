import numpy as np
import pandas as pd
import copy
import random

iterations = 10
SEED = 2023
random.seed(SEED)


def cross_validation(dataset, k):
    n = int(len(dataset) / k)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    for i in range(k):
        test_indices = np.random.choice(len(dataset), n, replace=False)
        test = dataset.iloc[test_indices].reset_index(drop=True)
        train = dataset.drop(test_indices).reset_index(drop=True)
    return train, test


def confusion_matrix(classes, predicted, expected):
    print('Confusion matrix!')
    matrix = np.zeros((len(classes), len(classes)))
    for i, j in zip(expected, predicted):
        #print(f"Expected: {i}, Predicted: {j}")
        matrix[get_index(i)][get_index(j)] += 1
    print(matrix)
    return matrix


def get_index(category):
    if category == 'Deportes':
        return 0
    if category == 'Destacadas':
        return 1
    if category == 'Nacional':
        return 2
    if category == 'Salud':
        return 3


def accurancy(confusion_matrix):
    correct = sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))])
    total = sum(sum(confusion_matrix))
    result = correct / total
    return result

