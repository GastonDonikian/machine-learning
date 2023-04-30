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



def random_forest(training,attributes,tree_count,max,filter_min_data,filter_gain):
    nodes_fathers =[]
    df_list = metrics.bagging(training,tree_count)
    
    for data_frame in df_list:
        values_per_atr = {}
        for atr in attributes :
            values_per_atr[atr] = data_frame[atr].unique()
        
        nodes_fathers.append(id3_algorithm.id3(data_frame,attributes,values_per_atr,None, max,filter_min_data,filter_gain))

    return nodes_fathers