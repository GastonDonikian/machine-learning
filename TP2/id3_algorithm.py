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



creditability = 'Creditability'
creditability_values = [0, 1]
total_nodes = 0


class Node:
    def __init__(self, desc_list, value, entropy, gain, attribute):
        self.descendants = desc_list
        self.entropy = entropy
        self.value = value
        self.gain = gain
        self.attribute = attribute
        self.moda = None

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

def finish_tree(training,node):
    crediatability_filter = training.loc[training[creditability] == 0]
    size_0 = crediatability_filter.shape[0]
    size_1 = 1 - training.shape[0]
    if size_0 > size_1:
        node_cred_0 =  Node([],0,None,None,creditability)
        if node is not None:
            node.moda = 0
        node.append_desc(node_cred_0)
    else:    
        node_cred_1 =  Node([],1,None,None,creditability)
        if node is not None:
            node.moda = 1
        node.append_desc(node_cred_1)


def check_tree(training, node, filter_min_data, filter_probability):
    crediatability_filter = training.loc[training[creditability] == 0]
    size_0 = crediatability_filter.shape[0]
    size_1 = training.shape[0] - size_0
    size = training.shape[0]
    prob_0 = 0
    prob_1 = 0
    if size != 0:
        prob_0 = size_0/size
        prob_1 = size_1/size

    if filter_min_data is not None and size <= filter_min_data:
        finish_tree(training,node)
        return node
        
    if size_0 == 0 or (filter_probability is not None and prob_1 >= filter_probability): 
        new_node = Node([],1,None,None,creditability)
        if node is not None:
            node.moda = 1
        return new_node
    elif size_0 == training.shape[0] or (filter_probability is not None and prob_0 >= filter_probability):
        new_node = Node([],0,None,None,creditability)
        if node is not None:
            node.moda = 0
        return new_node
    else:
        if node is not None:
            node.moda = 0 if size_0 > size_1 else 1 
        return None
    
def id3(training, attributes, values, father, max, filter_min_data , filter_gain, filter_probability, max_nodes):
    #global total_nodes

    if len(attributes) == 0 or len(values) == 0 or len(training) == 0:
        return father

    if father is None:
        father = Node([],None,None,None,None)
        n = check_tree(training, father, None, None)
        if n is not None:
            father.append_desc(n)
            return father
        
    if max is not None: max -= 1

    
    probability_dict_attribute,father_g = get_probabilities_and_father_g(training, attributes, values, father)
    attribute_max_gain, entropies, gain = calculate_max_gain_and_entropies(probability_dict_attribute,father_g,training.shape[0])

    if father is not None and father.value is not None and filter_gain is not None and gain <= filter_gain:
        finish_tree(training, father)
        return father
    

    
    max_childs = 0
    remainder = 0
    if max_nodes is not None: 
        if max_nodes <= 1:
            finish_tree(training, father)
            return father
        max_childs = max_nodes // len(values[attribute_max_gain])
        remainder = max_nodes % len(values[attribute_max_gain])
        
        

    for idx,value in enumerate(values[attribute_max_gain]):
        nodes_childs = None
        if max_nodes is not None:
            nodes_childs = 0  
            if remainder > 0:
                remainder -= 1
                nodes_childs = 1
            nodes_childs += max_childs
            if nodes_childs == 0:
                finish_tree(training, father)
                return father
            nodes_childs-=1

        new_node = Node([],value,entropies[idx],gain,attribute_max_gain)

        #total_nodes += 1

        father.append_desc(new_node)
        mask = training[attribute_max_gain] == value
        training_filter = training[mask]
        node_leaf = check_tree(training_filter, new_node, filter_min_data, filter_probability)
        if node_leaf is not None:
            new_node.append_desc(node_leaf)
        else:
            new_attributes = attributes.copy()
            new_attributes.remove(attribute_max_gain)
            new_val = values.copy()
            new_val.pop(attribute_max_gain)
        
            if (max is None and new_attributes is not None ) or ( max > 0):
                id3(training_filter,new_attributes,new_val,new_node,max,filter_min_data,filter_gain, filter_probability,nodes_childs)
            else:
                finish_tree(training, new_node)

    #print(total_nodes)
    return father

    