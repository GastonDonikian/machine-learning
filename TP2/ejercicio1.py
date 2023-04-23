import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import random
import metrics
import numpy as np
import math
from collections import defaultdict

creditability = "Creditability"

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
    def __init__(self, desc_list, value, entropy):
        self.descendants = desc_list
        self.entropy = entropy
        self.value = value

    def get_descendant(self, i: int):
        return self.descendants[i]
    
    def get_descendants(self):
        return self.descendants

    def get_value(self):
        return self.value
    
    def get_entropy(self):
        return self.entropy


def entropy(probabilities):
    en = 0
    for p in probabilities:
        if p != 0:
            en -= p * math.log2(p)
    return en
    

def gain(relative_probs_value, values,g):
    
    print("relative props")
    print(relative_probs_value)
    entropies = []
    for value in values:
        e= entropy(relative_probs_value.get(value)[1:])
        g -= relative_probs_value.get(value)[0] *  e
        entropies.append(e)
    
    return g, entropies
    

def set_probabilities(training,attr_name,values):
    relative_probs_value = {}
    for value in values:
        training_filter = training.loc[training[attr_name] == value]
        relative_probs_value[value] = []
        size = training.shape[0]
        if size == 0:
            relative_probs_value[value].append(0)
        else:
            relative_probs_value[value].append(((training_filter).shape[0])/size) 
        crediatability_filter = training_filter.loc[training[creditability] == 0]
        size = training_filter.shape[0]
        if size == 0:
            relative_probs_value[value].append(0)
            relative_probs_value[value].append(0)
        else: 
            relative_probs_value[value].append(crediatability_filter.shape[0]/size)
            relative_probs_value[value].append((training_filter.shape[0]-crediatability_filter.shape[0])/size)
    return relative_probs_value

def get_criteria_probabilities(data_array, value):
    return data_array.count(value)/len(data_array)


def get_max_gain(training, attributes, values, father):
    max_gain = 0
    attribute_max_gain = ""
    g = 0
    entropies = []
    if father is not None:
        g = father.get_entropy()
    else:
        probabilities = []
        crediatability_filter = training.loc[training[creditability] == 0]
        probabilities.append(crediatability_filter.shape[0]/training.shape[0])
        probabilities.append((training.shape[0]-crediatability_filter.shape[0])/training.shape[0])
        g = entropy(probabilities)


    for attribute in attributes:
        relative_probs_value = set_probabilities(training,attribute, values[attribute])
        g, e = gain(relative_probs_value,values[attribute], g)
        print(attribute)
        print(max_gain)
        print(g)
        if max_gain < g:
            max_gain = g
            attribute_max_gain = attribute
            entropies = e

    print("ATTRIBUTE MAX GAIN " + attribute_max_gain)
    return attribute_max_gain, entropies


def id3(training, attributes, values, father):
       
    if len(attributes) == 0:
        return father
    
    
    attribute_max_gain, entropies = get_max_gain(training, attributes, values, father)

    print("ID3")
    print(attribute_max_gain)
    print(entropies) ##ta dando raro esto
    for idx,value in enumerate(values[attribute_max_gain]):
        new_node = Node([],value,entropies[idx])
        if father is not None:
            father.descendants.append(new_node)
        training_filter = training.loc[training[attribute_max_gain] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(attribute_max_gain)
        new_val = values.copy()
        new_val.pop(attribute_max_gain)
        id3(training_filter,new_attributes,new_val,new_node)

    return father

    
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

    attributes.remove(creditability)
    print(attributes)
    print("------")
    values_per_atr = {}
    for atr in attributes :
        values_per_atr[atr] = data[atr].unique()


    df_list = metrics.cross_validation(data, 10)
    test = df_list[0]
    training = pd.DataFrame()
    for j in range(1, 10):
         training = pd.concat([training, df_list[j]], axis=0)

    print("MAIN")

    print(id3(data,attributes,values_per_atr,None))
    
