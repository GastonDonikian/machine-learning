from utils_id3 import calculate_max_gain_and_entropies

creditability = 'Creditability'
creditability_values = [0, 1]


class Node:
    def __init__(self, desc_list, value, gain, attribute):
        self.descendants = desc_list
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




def finish_tree(training,node):
    crediatability_filter = training.loc[training[creditability] == 0]
    size_0 = crediatability_filter.shape[0]
    size_1 = training.shape[0] - size_0
    if size_0 > size_1:
        node_cred_0 =  Node([],0,None,creditability)
        if node is not None:
            node.moda = 0
        node.append_desc(node_cred_0)
    else:    
        node_cred_1 =  Node([],1,None,creditability)
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
        new_node = Node([],1,None,creditability)
        if node is not None:
            node.moda = 1
        return new_node
    elif size_0 == training.shape[0] or (filter_probability is not None and prob_0 >= filter_probability):
        new_node = Node([],0,None,creditability)
        if node is not None:
            node.moda = 0
        return new_node
    else:
        if node is not None:
            node.moda = 0 if size_0 > size_1 else 1 
        return None

def count_nodes(node):
    if node is None or node.attribute == creditability:
        return 0
    total_nodes = 0
    for n in node.descendants:
        total_nodes += count_nodes(n)
    return total_nodes + 1
    
def id3(training, attributes, values, father, max, filter_min_data , filter_gain, filter_probability, max_nodes):

    if len(attributes) == 0 or len(values) == 0 or len(training) == 0:
        return father

    if father is None:
        father = Node([],None,None,None)
        n = check_tree(training, father, None, None)
        if n is not None:
            father.append_desc(n)
            return father
        
    if max is not None: max -= 1

    
    attribute_max_gain, gain = calculate_max_gain_and_entropies(training,attributes,values)

    if father is not None and father.value is not None and filter_gain is not None and gain <= filter_gain:
        finish_tree(training, father)
        return father
    

    
    max_childs = 0
    remainder = 0
    if max_nodes is not None: 
        if max_nodes < 1:
            finish_tree(training, father)
            return father
        max_childs = max_nodes // len(values[attribute_max_gain])
        remainder = max_nodes % len(values[attribute_max_gain])
        #print("MAX_NODES", max_nodes, "max_child", max_childs, "remainder", remainder, "values", len(values[attribute_max_gain]))
        

    #iterate through the value of the attribute that has max gain
    for idx,value in enumerate(values[attribute_max_gain]):
        #print("for iteration", idx, "max_childs", max_childs, "max_nodes", max_nodes)
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

        new_node = Node([],value, gain,attribute_max_gain)
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
                #print("calling id3", nodes_childs, "max childs", max_childs)
                id3(training_filter,new_attributes,new_val,new_node,max,filter_min_data,filter_gain, filter_probability,nodes_childs)
            else:
                finish_tree(training, new_node)

    return father

    