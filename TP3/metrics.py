import numpy as np
import pandas as pd
import copy
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

iterations = 10
SEED = 5017
random.seed(SEED)


def cross_validation(dataset, k,seed= SEED):
    np.random.shuffle(dataset)
    dataset = np.array_split(dataset, k)
    return dataset

def choose_test(index,dataset):
    test = dataset[index]
    dataset = np.delete(dataset,index)
    training = np.concatenate(dataset)
    return test, training 

def plot_confusion_matrix(title, conf_matrix):
    # Change figure size and increase dpi for better resolution
    plt.figure(figsize=(8, 6), dpi=100)
    # Scale up the size of all text
    sns.set(font_scale=1.1)

    # Plot Confusion Matrix using Seaborn heatmap()
    # Parameters:
    # first param - confusion matrix in array format
    # annot = True: show the numbers in each heatmap cell
    # fmt = 'd': show numbers as integers.
    ax = sns.heatmap(conf_matrix, annot=True, fmt='g')

    # set x-axis label and ticks.
    ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['Pasto', 'Vaca', 'Cielo'])

    # set y-axis label and ticks
    ax.set_ylabel("Actual", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['Pasto', 'Vaca', 'Cielo'])

    # set plot title
    ax.set_title(title + " confusion matrix", fontsize=14, pad=20)

    plt.savefig('./images/' + title + '_pasto_vaca_cielo_confusion_matrix.png', bbox_inches='tight')
    plt.show()


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


def confusion_matrix(classes, predicted, expected):
    print('Confusion matrix!')
    matrix = np.zeros((len(classes), len(classes)))
    for i, j in zip(expected, predicted):
        # print(f"Expected: {i}, Predicted: {j}")
        matrix[i][j] += 1
    return matrix


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
