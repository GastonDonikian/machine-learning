import pandas as pd
from line_profiler import LineProfiler
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
        mask = training[attr_name] == value
        training_filter = training[mask]
        relative_probs_value[value] = []
        mask = training_filter[creditability] == 0
        crediatability_filter = training_filter[mask]
        relative_probs_value[value].append(calculate_probability(training_filter.shape[0],training.shape[0]))
        relative_probs_value[value].append(calculate_probability(crediatability_filter.shape[0],training_filter.shape[0]))
        relative_probs_value[value].append(calculate_probability(training_filter.shape[0]-crediatability_filter.shape[0],training_filter.shape[0]))
    return relative_probs_value

def get_probabilities_and_father_g(training, attributes, values, father):
    max_gain = 0
    attribute_max_gain = ""
    entropies = []
    probability_dict_attribute = {item: {} for item in attributes} #the value is set an another dict

    creditability_0 = 0
    creditability_1 = 0

    father_g = 0
    
    for key in probability_dict_attribute:
        probability_dict_attribute[key] = {item: {} for item in values[key]}
        for key_value in probability_dict_attribute[key]:
            probability_dict_attribute[key][key_value] = {item: 0 for item in creditability_values}

    #one time iterete all over the dataframe        
    for index, row in training.iterrows():
        for column_name, column_data in row.items():
            if column_name in attributes:
                    probability_dict_attribute[column_name][column_data][row[creditability]] += 1 
            if column_name == creditability:
                if father is None or father.value is None:
                    creditability_0,creditability_1 = (creditability_0 + 1, creditability_1) if column_data == 0 else (creditability_0, creditability_1 + 1)

     
    if father is not None and father.value is not None:
        father_g = father.get_entropy()
    else:
        probabilities = []
        probabilities.append(creditability_0/(creditability_0 + creditability_1))
        probabilities.append(creditability_1/(creditability_1 + creditability_0))
        father_g = entropy(probabilities)


    return probability_dict_attribute, father_g


def calculate_max_gain_and_entropies(probability_dict_attribute, father_g, total):
    max_gain = 0
    attribute_max_gain = None
    entropies = []
    for key in probability_dict_attribute:
       probabilities_dict = {}
       for key_value in probability_dict_attribute[key]:
            creditability_0 = 0
            creditability_1 = 0
            creditability_0 = probability_dict_attribute[key][key_value][0]
            creditability_1 = probability_dict_attribute[key][key_value][1]
            probabilities = []
            creditability_sum = creditability_0 + creditability_1
            if creditability_sum != 0:
                probabilities.append(creditability_sum/total)
                probabilities.append(creditability_0/creditability_sum)
                probabilities.append(creditability_1/creditability_sum)
            else: #Si el atributo da 0?
                probabilities.append(creditability_sum/total)
                probabilities.append(creditability_sum)
                probabilities.append(creditability_sum)

            probabilities_dict[key_value] = probabilities
    g, e = gain(probabilities_dict, probability_dict_attribute[key].keys(), father_g)
    if max_gain <= g:
        max_gain = g
        attribute_max_gain = key
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
        gain, entropies = calculate_gain_and_entropy(training,creditability,creditability_values,father)
        node_cred_0 =  Node([],0,entropies[0],gain,creditability)
        node.append_desc(node_cred_0)
        node_cred_1 =  Node([],1,entropies[0],gain,creditability)
        node.append_desc(node_cred_1)


def check_tree(training):
    crediatability_filter = training.loc[training[creditability] == 0]
    size = crediatability_filter.shape[0]
    if size == 0: 
        new_node = Node([],1,None,None,creditability)
        return new_node
    elif size == training.shape[0]:
        new_node = Node([],0,None,None,creditability)
        return new_node
    else:
        return None
    
def id3(training, attributes, values, father, max):
       
    if len(attributes) == 0 or len(values) == 0 or len(training) == 0:
        return father

    if father is None:
        n = check_tree(training)
        if n is not None:
            return n
        father = Node([],None,None,None,None)
    if max is not None: max -= 1
    
    probability_dict_attribute,father_g = get_probabilities_and_father_g(training, attributes, values, father)
    attribute_max_gain, entropies, gain = calculate_max_gain_and_entropies(probability_dict_attribute,father_g,training.shape[0])
    for idx,value in enumerate(values[attribute_max_gain]):
        new_node = Node([],value,entropies[idx],gain,attribute_max_gain)
        training_filter = training.loc[training[attribute_max_gain] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(attribute_max_gain)
        new_val = values.copy()
        new_val.pop(attribute_max_gain)
        father.append_desc(new_node)
        if (max is None and new_attributes is not None ) or ( max > 0):
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


def main():
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


    father = id3(data,attributes,values_per_atr,None,2)
    print(father)

if __name__ == "__main__":
    main() 
