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


def confusion_matrix_by_category(category, predicted, expected):
    matrix = np.zeros((2, 2))
    for i, j in zip(expected, predicted):
        if i == j:
            if category == i:
                matrix[0][0] += 1
            else:
                matrix[1][1] += 1
        else:
            if i == category:
                matrix[0][1] += 1
            else:
                matrix[1][0] += 1
    return matrix


def confusion_matrix(classes, predicted, expected):
    print('Confusion matrix!')
    matrix = np.zeros((len(classes), len(classes)))
    for i, j in zip(expected, predicted):
        # print(f"Expected: {i}, Predicted: {j}")
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

def precision(confusion_matrix):
    TP = confusion_matrix[0,0]
    FP = confusion_matrix[1,0]
    result =  TP / (TP + FP)
    return result

def recall(confusion_matrix):
    TP = confusion_matrix[0,0]
    FN = confusion_matrix[0,1]
    recall = TP / (TP + FN)
    return recall


def F1_score(confusion_matrix):
    r = recall(confusion_matrix)
    p = precision(confusion_matrix)
    result = 1*p*r / (p + r)
    return result

