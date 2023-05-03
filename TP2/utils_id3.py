import math
import numpy

creditability = 'Creditability'
creditability_values = [0, 1]




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

def get_probabilities(training, attributes, values):
    probability_dict_attribute = {item: {} for item in attributes} #the value is set an another dict

    creditability_0 = 0
    creditability_1 = 0
    
    for attr in probability_dict_attribute:
        probability_dict_attribute[attr] = {item: {} for item in values[attr]}
        for value in probability_dict_attribute[attr]:
            probability_dict_attribute[attr][value] = {item: 0 for item in creditability_values}

    #one time iterete all over the dataframe        
    for index, row in training.iterrows():
        for column_name, column_data in row.items():
            if column_name in attributes:
                    probability_dict_attribute[column_name][column_data][row[creditability]] += 1 
            if column_name == creditability:
                creditability_0,creditability_1 = (creditability_0 + 1, creditability_1) if column_data == 0 else (creditability_0, creditability_1 + 1)

  
    probabilities = []
    probabilities.append(creditability_0/(creditability_0 + creditability_1))
    probabilities.append(creditability_1/(creditability_1 + creditability_0))
    entropy_general = entropy(probabilities)


    return probability_dict_attribute,  entropy_general


def calculate_gains(training, attributes, values):
    probability_dict_attribute, entropy_general = get_probabilities(training, attributes, values)
    gains = {}
    total = training.shape[0]
    for attr in probability_dict_attribute:
        gain = entropy_general
        for value in probability_dict_attribute[attr]:
            creditability_0 = probability_dict_attribute[attr][value][0]
            creditability_1 = probability_dict_attribute[attr][value][1]
            creditability_sum = creditability_0 + creditability_1
            if creditability_sum != 0:
                probs = []
                probs.append(creditability_0/creditability_sum)
                probs.append(creditability_1/creditability_sum)
                gain -= creditability_sum/total * entropy(probs)
            
        gains[attr] = gain
    
    return gains

def calculate_max_gain_and_entropies(training,attributes, values):
    probability_dict_attribute, entropy_general = get_probabilities(training, attributes, values)
    max_gain = 0
    attribute_max_gain = None
    total = training.shape[0]
    for attr in probability_dict_attribute:
        gain = entropy_general
        for value in probability_dict_attribute[attr]:
            creditability_0 = probability_dict_attribute[attr][value][0]
            creditability_1 = probability_dict_attribute[attr][value][1]
            creditability_sum = creditability_0 + creditability_1
            if creditability_sum != 0:
                probs = []
                probs.append(creditability_0/creditability_sum)
                probs.append(creditability_1/creditability_sum)
                gain -= creditability_sum/total * entropy(probs)
        
        if gain >= max_gain:
            max_gain = gain
            attribute_max_gain = attr

    
    return attribute_max_gain, max_gain