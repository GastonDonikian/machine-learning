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
    def __init__(self, desc_list, value, entropy, gain, attribute):
        self.descendants = desc_list
        self.entropy = entropy
        self.value = value
        self.gain = gain
        self.attribute = attribute

    def get_descendant(self, i: int):
        return self.descendants[i]
    
    def get_descendants(self):
        return self.descendants

    def get_value(self):
        return self.value
    
    def get_entropy(self):
        return self.entropy
    
    def get_attribute(self):
        return self.attribute
    
    def append_desc(self, node):
        self.descendants.append(node)

    def get_gain(self):
        return self.gain
    
    def __repr__(self):
        return "desc_list:% s value:% s Attribute:% s" % (self.descendants, self.value, self.attribute) 

    def __str__(self): 
        return "desc_list:% s value:% s Attribute:% s" % (self.descendants, self.value, self.attribute) 


def entropy(probabilities):
    en = 0
    for p in probabilities:
        if p != 0:
            en -= p * math.log2(p)
    return en
    

def gain(relative_probs_value, values,g):
    entropies = []
    for value in values:
        e= entropy(relative_probs_value.get(value)[1:])
        g -= relative_probs_value.get(value)[0] *  e
        entropies.append(e)
    
    return g, entropies
    

def calculate_probability(training_filter,training):
      if training == 0:
          return 0
      else:
          return training_filter/training


def set_probabilities(training,attr_name,values):
    relative_probs_value = {}
    for value in values:
        training_filter = training.loc[training[attr_name] == value]
        relative_probs_value[value] = []
        crediatability_filter = training_filter.loc[training[creditability] == 0]
        relative_probs_value[value].append(calculate_probability(training_filter.shape[0],training.shape[0]))
        relative_probs_value[value].append(calculate_probability(crediatability_filter.shape[0],training_filter.shape[0]))
        relative_probs_value[value].append(calculate_probability(training_filter.shape[0]-crediatability_filter.shape[0],training_filter.shape[0]))
    return relative_probs_value


#@profile
def get_max_gain(training, attributes, values, father):
    max_gain = 0
    attribute_max_gain = ""
    entropies = []

    for attribute in attributes:
        relative_probs_value = set_probabilities(training,attribute, values[attribute])
        g, e = calculate_gain_and_entropy(training,attribute,values[attribute],father)
        if max_gain <= g:
            max_gain = g
            attribute_max_gain = attribute
            entropies = e  
    return attribute_max_gain, entropies, max_gain


def calculate_gain_and_entropy(training,attribute,value,father):
    
    if father is not None and father.value is not None:
        father_g = father.get_entropy()
    else:
        probabilities = []
        crediatability_filter = training.loc[training[creditability] == 0]
        probabilities.append(crediatability_filter.shape[0]/training.shape[0])
        probabilities.append((training.shape[0]-crediatability_filter.shape[0])/training.shape[0])
        father_g = entropy(probabilities)

    relative_probs_value = set_probabilities(training,attribute, value)
    g, e = gain(relative_probs_value,value, father_g)

    return g, e

def finish_tree(training,node,father):
    new_node = check_tree(training)
    if new_node is not None:
        node.append_desc(new_node)
    else:
        gain, entropies = calculate_gain_and_entropy(training,creditability,["0","1"],father)
        node_cred_0 =  Node([],"0",entropies[0],gain,creditability)
        node.append_desc(node_cred_0)
        node_cred_1 =  Node([],"1",entropies[0],gain,creditability)
        node.append_desc(node_cred_1)


#@profile
def check_tree(training):
    crediatability_filter = training.loc[training[creditability] == 0]
    size = crediatability_filter.shape[0]
    if size == 0: 
        new_node = Node([],"1",None,None,creditability)
        return new_node
    elif size == training.shape[0]:
        new_node = Node([],"0",None,None,creditability)
        return new_node
    else:
        return None
    

#@profile
def id3(training, attributes, values, father, max):
       
    if len(attributes) == 0 or len(values) == 0 or len(training) == 0:
        return father

    if father is None:
        n = check_tree(training)
        if n is not None:
            return n
        father = Node([],None,None,None,None)
    max -= 1
    attribute_max_gain, entropies, gain = get_max_gain(training, attributes, values, father)
    for idx,value in enumerate(values[attribute_max_gain]):
        new_node = Node([],value,entropies[idx],gain,attribute_max_gain)
        training_filter = training.loc[training[attribute_max_gain] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(attribute_max_gain)
        new_val = values.copy()
        new_val.pop(attribute_max_gain)
        father.append_desc(new_node)
        if max > 0:
            id3(training_filter,new_attributes,new_val,new_node,max)
        else:
            finish_tree(training,new_node,father)
        


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
    values_per_atr = {}
    for atr in attributes :
        values_per_atr[atr] = data[atr].unique()


    df_list = metrics.cross_validation(data, 10)
    test = df_list[0]
    training = pd.DataFrame()
    for j in range(1, 10):
         training = pd.concat([training, df_list[j]], axis=0)


    print(id3(data,attributes,values_per_atr,None,2))
    
