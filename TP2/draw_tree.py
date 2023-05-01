from graphviz import Digraph
import matplotlib.pyplot as plt
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
from collections import defaultdict


def graph_tree(root):
    def add_nodes_edges(node, dot=None):
        if dot is None:
            dot = Digraph()
            dot.attr('node', shape='oval')
            dot.node(str(id(node)), label=str(node.attribute))
        for child in node.descendants:
            if child:
                if True:
                    dot.attr('node', shape='oval')
                    dot.node(str(id(child)), label=str(child.attribute + " = " + str(child.value)))
                    dot.edge(str(id(node)), str(id(child)))
                else:
                    dot.attr('node', shape='box')
                    dot.node(str(child), label=str(child))
                    dot.edge(str(node), str(child))
                add_nodes_edges(child, dot)
        return dot

    dot = add_nodes_edges(root)
    #print(dot.source)
    dot.render('tree', view=True)