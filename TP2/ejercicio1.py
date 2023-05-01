import pandas as pd
from line_profiler import LineProfiler
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import random
import metrics
import numpy as np
import math
import id3_algorithm
import randomForest
import matplotlib.pyplot as plt
import draw_tree
from collections import defaultdict

creditability = 'Creditability'
creditability_values = [0, 1]

def plot_gain(data_frame, attributes):

    for attribute in attributes:
        gain = id3_algorithm.gain()

def preprocessing(data_frame, name, parts):
    data = data_frame[name].to_numpy()
    data = np.sort(data)
    size = len(data)
    res = []
    partition = int(size/parts)
    for i in range(parts - 1):
        res.append(data[(i+1)*partition])
    return res



    
def get_nearest_index(value, processed_array):
    index = 0
    for i in processed_array:
        if i >= value:
            return index
        index += 1

    return index

def replaces_process_data(data, name, parts):
    processed_array = preprocessing(data,name , parts)
    result = list(
        map(
        lambda x: get_nearest_index(x, processed_array), data[name].values))
    data[name] = result
    return data


def classify_input(row, father):

    while True:
        change = False
        desc = father.descendants
        for node in desc:
            if node.attribute == creditability:
                    clasify = node.value
                    return clasify
            else:
                if node.value == row[node.attribute]:
                    father = node
                    change = True

        if change == False:
            classify = father.moda
            return classify




def expected(test):
    expected = []

    for index, row in test.iterrows():
        expected.append(row[creditability])
    
    return expected

def predicted(test, father):
    predicted_result = []

    for index, row in test.iterrows():
        classify = classify_input(row,father)
        predicted_result.append(classify)

    return predicted_result


def resolve_test(test,father):
    expected_result = []
    predicted_result = []

    for index, row in test.iterrows():
        expected_result.append(row[creditability])
        predicted_result.append(classify_input(row,father))
        
    confusion_matrix, tasa_falsos_positivos, tasa_verdaderos_postivos = metrics.confusion_matrix_by_category(creditability_values[1], expected_result, predicted_result)
    print(confusion_matrix)
    print(tasa_falsos_positivos)
    print(tasa_verdaderos_postivos)
    print(metrics.accuracy(confusion_matrix))
    return metrics.accuracy(confusion_matrix)
    

def resolve_random_forest(dict_predicted, expected_results):
    predicted = []

    for index, value in enumerate(expected_results):
        result = dict_predicted[index]
        count_0 = result[0]
        count_1 = result[1]
        if count_1 > count_0:
            predicted.append(1)
        else:
            predicted.append(0)
    
    confusion_matrix, tasa_falsos_positivos, tasa_verdaderos_postivos = metrics.confusion_matrix_by_category(creditability_values[1], expected_results, predicted)
    print(confusion_matrix)
    print(tasa_falsos_positivos)
    print(tasa_verdaderos_postivos)
    print(metrics.accuracy(confusion_matrix))
    return metrics.accuracy(confusion_matrix)

def plot_max_nodes_precision(training, test, attributes):
    values_per_atr = {}
    for atr in attributes : 
        values_per_atr[atr] = training[atr].unique()

    dict_depth = {}
    for i in range(180, 1880, 100):
        father = id3_algorithm.id3(training,attributes,values_per_atr,None, None,None,None,None,i) #tree of the training
        accuracy = resolve_test(test, father)
        dict_depth[i] = accuracy
    

    fig, ax = plt.subplots()

    # set the x-axis to logarithmic scale
    #ax.set_xscale('log')
    x = []
    y = []
    for key, value in sorted(dict_depth.items()):
        x.append(key)
        y.append(value)
        ax.scatter(key, value, color='blue')


    # join the points with a line
    ax.plot(x, y, color='blue')
    plt.show()


def main():
    data = pd.read_csv('./resources/german_credit.csv')
    attributes = list(data.head(0))
    data = replaces_process_data(data, 'Duration of Credit (month)', 3)
    data = replaces_process_data(data, 'Credit Amount', 4)
    data = replaces_process_data(data, 'Age (years)', 5)

   
    partition = 5
    df_list = metrics.cross_validation(data, partition)
    test = df_list[0]
    training = pd.DataFrame()
    for j in range(1, partition):
         training = pd.concat([training, df_list[j]], axis=0)

    attributes.remove(creditability) 


    #########################################################
    #execute ID3
    values_per_atr = {}
    for atr in attributes : 
       values_per_atr[atr] = training[atr].unique()
    father = id3_algorithm.id3(training,attributes,values_per_atr,None,None, None,None,None,12) #tree of the training
    resolve_test(test, father)
    draw_tree.graph_tree(father)
    print(id3_algorithm.count_nodes(father))

    #########################################################
    #execute Random forest
    #fathers = randomForest.random_forest(training,attributes,10,None,None,None,0.7)
    #dict_predicted = {}
    #for index, row in test.iterrows():
    #    dict_predicted[index] = {item: 0 for item in creditability_values}

    #for father in fathers:
    #    pred = predicted(test, father)

    #    for index, value in enumerate(pred):
    #        dict_predicted[index][value] += 1

    #expected_result = expected(test)

    #resolve_random_forest(dict_predicted,expected_result)


    #plot_max_nodes_precision(training, test, attributes) 


   
if __name__ == "__main__":
    main() 
