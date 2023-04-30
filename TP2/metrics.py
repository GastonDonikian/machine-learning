import numpy as np
import pandas as pd
import copy
import random
import matplotlib.pyplot as plt

iterations = 10


# SEED = 2023
# random.seed(SEED)


# def cross_validation(dataset, k):
#    n = int(len(dataset) / k)
#    dataset = dataset.sample(frac=1).reset_index(drop=True)
#    for i in range(k):
#        test_indices = np.random.choice(len(dataset), n, replace=False)
#        test = dataset.iloc[test_indices].reset_index(drop=True)
#        train = dataset.drop(test_indices).reset_index(drop=True)
#    return train, test

def cross_validation(dataset, k):
    # dataset = dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)
    dataset = dataset.dropna()
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    df_list = np.array_split(dataset, k)
    return df_list


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
    denominator = matrix[0][0] + matrix[0][1]
    tasa_falsos_positivos = 0
    tasa_verdaderos_postivos = 0
    if denominator != 0:
        tasa_falsos_positivos = matrix[0][1] / (denominator)
        tasa_verdaderos_postivos = matrix[0][0] / (denominator)

    return matrix, tasa_falsos_positivos, tasa_verdaderos_postivos


def calculate_accuracy(expected, predicted):
    count = 0
    for i, j in zip(expected, predicted):
        if i == j:
            count += 1
    return count / len(expected)


def plot_roc(individual_classifications: dict):
    actual = individual_classifications['Actual']  # expected

    individual_classifications.pop('Actual')
    roc_ratios = {}
    total_count = len(actual)
    total_count_positive = {}
    for k in individual_classifications.keys():
        roc_ratios[k] = []
        total_count_positive[k] = len(list(filter(lambda p: p == k, actual)))
    print(total_count)
    for threshold in np.linspace(start=0, stop=1, num=10):
        for classification in individual_classifications.keys():
            true_positive = 0
            false_positive = 0
            for i, j in zip(actual, individual_classifications[classification]):
                if j > float(threshold):
                    if classification == i:
                        true_positive += 1
                    else:
                        false_positive += 1
            roc_ratios[classification].append((false_positive / (total_count - total_count_positive[classification]),
                                               true_positive / total_count_positive[classification]))
    for k in roc_ratios.keys():
        plt.plot(*zip(*roc_ratios[k]), label=k)
    plt.title('ROC Curve')
    plt.plot(np.linspace(start=0, stop=1, num=10), np.linspace(start=0, stop=1, num=10), label='y = x',
             linestyle='dashed', linewidth=0.9)
    plt.legend()
    plt.xlabel('False positives')
    plt.ylabel('True positives')
    plt.show()


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
    if category == 'Economia':
        return 1
    if category == 'Nacional':
        return 2
    if category == 'Salud':
        return 3


def accuracy(confusion_matrix):
    correct = sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))])
    total = sum(sum(confusion_matrix))
    result = correct / total
    return result


def precision(confusion_matrix):
    TP = confusion_matrix[0, 0]
    FP = confusion_matrix[1, 0]
    result = TP / (TP + FP)
    return result


def recall(confusion_matrix):
    TP = confusion_matrix[0, 0]
    FN = confusion_matrix[0, 1]
    recall = TP / (TP + FN)
    return recall


def F1_score(confusion_matrix):
    r = recall(confusion_matrix)
    p = precision(confusion_matrix)
    result = 2 * p * r / (p + r)
    return result
