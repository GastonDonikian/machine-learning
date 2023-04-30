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
from collections import defaultdict

creditability = 'Creditability'
creditability_values = [0, 1]



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
            return  father.moda


def resolve_test(test,father):
    expected = []
    predicted = []

    for index, row in test.iterrows():
        expected.append(row[creditability])
        predicted.append(classify_input(row,father))
        
    confusion_matrix, tasa_falsos_positivos, tasa_verdaderos_postivos = metrics.confusion_matrix_by_category(creditability_values[1], expected, predicted)
    print(confusion_matrix)
    print(tasa_falsos_positivos)
    print(tasa_verdaderos_postivos)
    print(metrics.accuracy(confusion_matrix))
    


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
    values_per_atr = {}
    for atr in attributes :
        values_per_atr[atr] = training[atr].unique()


    father = id3_algorithm.id3(training,attributes,values_per_atr,None, None,None,None) #tree of the training
 
    resolve_test(test, father)
   
if __name__ == "__main__":
    main() 
