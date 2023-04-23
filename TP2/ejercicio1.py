import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import random
import metrics
import numpy as np
import math
from collections import defaultdict


def preprocessing(data_frame, name, parts):
    data = data_frame[name].to_numpy()
    data = np.sort(data)
    size = len(data)
    res = []
    partition = int(size/parts)
    for i in range(parts - 1):
        res.append(data[(i+1)*partition])
    return res


class Node:
    def __init__(self, desc_list, value, data_filter):
        self.descendants = desc_list
        self.value = value
        self.data_filter = data_filter ##Esta es la data filtrada con todos los registros que tengan al padre y a la data filtrada de sus padres

    def get_descendant(self, i: int):
        return self.descendants[i]
    
    def get_descendants(self):
        return self.descendants

    def get_value(self):
        return self.value


def entropy(probabilities):
    en = 0
    for p in probabilities:
        en -= p * math.log2(p)
    return en
    

def gain(relative_probs_father_value,values,probabilities_father,probabilities_father_value):
    g = entropy(probabilities_father)
    
    for idx in range(values):
        g -= relative_probs_father_value[idx] *  entropy(probabilities_father_value[idx])
    
    return g
    

def set_probabilities(training_set,attr_name,father,values):
    relative_probs_father_value = []
    son_registers = []
    #probabilities_father_value = []
    #father_registers = filter()

    for v in values:
        relative_probs_father_value.append((father.registers.loc[father.registers[attr_name] == v].size[0]) /len(father.registers))
    return relative_probs_father_value

def get_criteria_probabilities(data_array, value):
    return data_array.count(value)/len(data_array)


def get_max_gain(training_set, attributes, values, father):
    max_gain = 0
    for attribute in attributes:
        relative_probs_father_value,probabilities_father_value = set_probabilities(training_set,attribute,data,father, values[attribute])
        g = gain(relative_probs_father_value,probabilities_father_value)
        max_gain = max(max_gain,max)

    return max(map(lambda x: gain(),training_set))


def id3(training_set, attributes, values, node):
    atribute_max_gain = get_max_gain(training_set, attributes, values, None)
    

    ## nodo vacio
    ##ganacia para los attributos set
    ##> ganancia --> lo agregamos al arbol
    ##volver a llamar a id3 pero sin ese atributo en el training_ set

    

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

if __name__ == "__main__":
    data = pd.read_csv('./resources/german_credit.csv')
    attributes = list(data.head(0))
    data = replaces_process_data(data, 'Duration of Credit (month)', 3)
    data = replaces_process_data(data, 'Credit Amount', 3)
    data = replaces_process_data(data, 'Age (years)', 3)

    values_per_atr = {}
    for atr in attributes :
        values_per_atr[atr] = data[atr].unique()


    df_list = metrics.cross_validation(data, 10)
    test = df_list[0]
    training = pd.DataFrame()
    for j in range(1, 10):
         training = pd.concat([training, df_list[j]], axis=0)

    
