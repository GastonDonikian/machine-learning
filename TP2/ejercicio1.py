import pandas as pd
from line_profiler import LineProfiler
from sklearn.model_selection import train_test_split
from collections import Counter
import metrics
import numpy as np
import id3_algorithm
import randomForest
import matplotlib.pyplot as plt
import draw_tree
from collections import defaultdict
import random
import pandas as pd
import utils_id3
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.ticker as ticker

creditability = 'Creditability'
creditability_values = [0, 1]


def plot_Random_forest_probability(data):

    partition = 5

    training = pd.DataFrame()
    attributes = list(data.head(0))
    attributes.remove(creditability) 

    
    repetitions = 5
    dict_depth = {}
    dict_depth_training = {}
    std_test = []
    std_training = []
    for i in range(0, 11):
        f = float(i) / 10.0
        acc_test = []
        acc_training = []
        for r in range(repetitions):    
            #DIVIDO TRAINING Y TESTEO
            df_list = metrics.cross_validation(data, partition,r)
            test = df_list[0]
            training = pd.DataFrame()
            for j in range(1, partition):
                training = pd.concat([training, df_list[j]], axis=0)
            
            #ejecuto random forest
            fathers = randomForest.random_forest(training,attributes,5,None,None,None,f,None,r)

            accuracy_test = accuracy_random_forest(fathers, test)
            accuracy_training = accuracy_random_forest(fathers, training)
            
           
            acc_test.append(accuracy_test)
            acc_training.append(accuracy_training)
           
        
        std_test.append( np.std(acc_test)/np.sqrt(repetitions))
        dict_depth[f] = np.mean(acc_test)
        std_training.append( np.std(acc_training)/np.sqrt(repetitions))
        
        dict_depth_training[f] = np.mean(acc_training)

    fig, ax = plt.subplots()

    # join the points with a line
    x = dict_depth.keys()
    y = dict_depth.values()
    ax.plot(x,y, ".-", color='cyan' )
    plt.errorbar(x,y, yerr = std_test ,capsize=2, elinewidth=0.5, label="Test")
    


    # join the points with a line
    x = dict_depth_training.keys()
    y = dict_depth_training.values()
    ax.plot(x,y,".-", color='pink')


    plt.errorbar(x,y, yerr = std_training, capsize=2, elinewidth=0.5, label="Training")
    plt.xlabel('Probability filter')
    plt.ylabel('Tree precision')
    plt.legend(loc='upper right')
    plt.show()


def plot_probability_precision(data):

    partition = 5

    training = pd.DataFrame()
    attributes = list(data.head(0))
    attributes.remove(creditability) 

    
    repetitions = 5
    dict_depth = {}
    dict_depth_training = {}
    std_test = []
    std_training = []
    for i in range(0, 11):
        f = float(i) / 10.0
        acc_test = []
        acc_training = []
        for r in range(repetitions):    
            df_list = metrics.cross_validation(data, partition,r)
            test = df_list[0]
            training = pd.DataFrame()
            for j in range(1, partition):
                training = pd.concat([training, df_list[j]], axis=0)
            values_per_atr = {}
            for atr in attributes : 
                values_per_atr[atr] = training[atr].unique()
            father = id3_algorithm.id3(training,attributes,values_per_atr,None, None,None,None,f,None) #tree of the training
            accuracy = resolve_test(test, father)
            #dict_depth[i] = accuracy
            accuracy_training = resolve_test(training, father)
            acc_test.append(accuracy)
            acc_training.append(accuracy_training)

            nodesCount = id3_algorithm.count_nodes(father)
            #dict_depth_training[i] = accuracy_training
        
        std_test.append( np.std(acc_test)/np.sqrt(repetitions))
       
        dict_depth[f] = np.mean(acc_test)
        std_training.append( np.std(acc_training)/np.sqrt(repetitions))
        dict_depth_training[f] = np.mean(acc_training)

    fig, ax = plt.subplots()


    # join the points with a line
    x = dict_depth.keys()
    y = dict_depth.values()
    ax.plot(x,y, ".-", color='cyan' )
    plt.errorbar(x,y, yerr = std_test ,capsize=2, elinewidth=0.5, label="Test")
    


    # join the points with a line
    x = dict_depth_training.keys()
    y = dict_depth_training.values()
    ax.plot(x,y,".-", color='pink')


    plt.errorbar(x,y, yerr = std_training, capsize=2, elinewidth=0.5, label="Training")
    plt.xlabel('Probability filter')
    plt.ylabel('Tree precision')
    plt.legend(loc='upper right')
    plt.show()


def plot_variables_count(data_frame):
    attribute = 'Duration of Credit (month)'
    values = data_frame[attribute].unique()

    plt.xlabel(attribute,fontsize=8)
    plt.ylabel('Count',fontsize=8)
    count = []
    for value in values:
        mask = data_frame[attribute] == value
        data_frame_filter = data_frame[mask]
        count.append(data_frame_filter.shape[0])

    # Create bars
    plt.bar(values,count, color=mcolors.TABLEAU_COLORS)
    # Show graph
    plt.show()


def plot_gain_number_values(data_frame):
    attributes = ['Duration of Credit (month)','Credit Amount',  'Age (years)']
    gains = {item: {} for item in attributes}
    fig, ax = plt.subplots()
    
    for i in range(2,10):
        df = data_frame.copy()
        df = replaces_process_data(df, 'Duration of Credit (month)', i)
        df = replaces_process_data(df, 'Credit Amount', i)
        df = replaces_process_data(df, 'Age (years)', i)
        values = calculate_values(attributes,df)
        gain= utils_id3.calculate_gains(df, attributes, values)
        for atr in gain.keys():
            gains[atr][i] = gain[atr] 

    
    ax.plot(gains['Duration of Credit (month)'].keys(), gains['Duration of Credit (month)'].values(), ".-", color='cyan',label ='Duration of Credit (month)' )
    ax.plot(gains['Credit Amount'].keys(), gains['Credit Amount'].values(), ".-", color='pink',label ='Credit Amount' )
    ax.plot(gains['Age (years)'].keys(), gains['Age (years)'].values(), ".-", color='green',label ='Age (years)' )

    
    #plt.legend( 'Duration of Credit (month)','Credit Amount',  'Age (years)')

    # Set the legend location and add the legend
    # Add legend
    plt.legend(loc='upper right')
    plt.xlabel('Number of values')
    plt.ylabel('Gain')

    plt.show()



def calculate_values(attributes, training):
    values_per_atr = {}
    for atr in attributes : 
        values_per_atr[atr] = training[atr].unique()
    
    return values_per_atr

def plot_gain(data_frame, attributes):
    attributes.remove(creditability) 
    values = calculate_values(attributes,data_frame)
    gains = utils_id3.calculate_gains(data_frame, attributes, values)
    gains = dict(sorted(gains.items(), key=lambda x: x[1]))


    plt.xlabel('Gain',fontsize=8)
    plt.ylabel('Attributes',fontsize=8)
    # Create bars
    plt.barh(list(gains.keys()),list(gains.values()), color=mcolors.TABLEAU_COLORS)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=6)
    # Show graph
    plt.show()

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
            classify = father.moda
            return classify




def expected(test):
    expected = []

    for index, row in test.iterrows():
        expected.append(row[creditability])
    
    return expected

def predicted(test, father):
    predicted_result = []

    for index, row in test.iterrows():
        classify = classify_input(row,father)
        predicted_result.append(classify)

    return predicted_result


def resolve_test(test,father):
    expected_result = []
    predicted_result = []

    for index, row in test.iterrows():
        expected_result.append(row[creditability])
        predicted_result.append(classify_input(row,father))
        
    confusion_matrix, tasa_falsos_positivos, tasa_verdaderos_postivos = metrics.confusion_matrix_by_category(creditability_values[0], expected_result, predicted_result)
    print(confusion_matrix)
    print(tasa_falsos_positivos)
    print(tasa_verdaderos_postivos)
    print(metrics.accuracy(confusion_matrix))

    heatmap_matrix(confusion_matrix)
    return metrics.accuracy(confusion_matrix)
    

def resolve_random_forest(dict_predicted, expected_results):
    predicted = []

    for index, value in enumerate(expected_results):
        result = dict_predicted[index]
        count_0 = result[0]
        count_1 = result[1]
        if count_1 > count_0:
            predicted.append(1)
        else:
            predicted.append(0)
    
    confusion_matrix, tasa_falsos_positivos, tasa_verdaderos_postivos = metrics.confusion_matrix_by_category(creditability_values[0], expected_results, predicted)
    print(confusion_matrix)
    print(tasa_falsos_positivos)
    print(tasa_verdaderos_postivos)
    print(metrics.accuracy(confusion_matrix))
    heatmap_matrix(confusion_matrix)
    return metrics.accuracy(confusion_matrix)


def plot_max_nodes_precision(data):

    partition = 5

    training = pd.DataFrame()
    attributes = list(data.head(0))
    attributes.remove(creditability) 

    
    repetitions = 5
    dict_depth = {}
    dict_depth_training = {}
    std_test = []
    std_training = []
    for i in range(100, 1000, 100):
        acc_test = []
        acc_training = []
        for r in range(repetitions):    
            df_list = metrics.cross_validation(data, partition,r)
            test = df_list[0]
            training = pd.DataFrame()
            for j in range(1, partition):
                training = pd.concat([training, df_list[j]], axis=0)
            values_per_atr = {}
            for atr in attributes : 
                values_per_atr[atr] = training[atr].unique()
            father = id3_algorithm.id3(training,attributes,values_per_atr,None, None,None,None,None,i) #tree of the training
            accuracy = resolve_test(test, father)
            #dict_depth[i] = accuracy
            accuracy_training = resolve_test(training, father)
            acc_test.append(accuracy)
            acc_training.append(accuracy_training)

            nodesCount = id3_algorithm.count_nodes(father)
            #dict_depth_training[i] = accuracy_training
        
        std_test.append( np.std(acc_test)/np.sqrt(repetitions))
       
        dict_depth[i] = np.mean(acc_test)
        std_training.append( np.std(acc_training)/np.sqrt(repetitions))
        dict_depth_training[i] = np.mean(acc_training)

    fig, ax = plt.subplots()


    # join the points with a line
    x = dict_depth.keys()
    y = dict_depth.values()
    ax.plot(x,y, ".-", color='cyan' )
    plt.errorbar(x,y, yerr = std_test ,capsize=2, elinewidth=0.5, label="Test")
    


    # join the points with a line
    x = dict_depth_training.keys()
    y = dict_depth_training.values()
    ax.plot(x,y,".-", color='pink')


    plt.errorbar(x,y, yerr = std_training, capsize=2, elinewidth=0.5, label="Training")
    plt.xlabel('Max Nodes')
    plt.ylabel('Tree precision')
    plt.legend(loc='upper right')
    plt.show()

def accuracy_random_forest(fathers, test):
    dict_predicted = {}
    test = test.reset_index()
    for index, row in test.iterrows():
        dict_predicted[index] = {item: 0 for item in creditability_values}
    
    for father in fathers:
        pred = predicted(test, father)
        for index, value in enumerate(pred):
            dict_predicted[index][value] += 1

    expected_result_test = expected(test)

    accuracy_test = resolve_random_forest(dict_predicted,expected_result_test)

    return accuracy_test

def plot_Random_forest_max_nodes(data):

    partition = 5

    training = pd.DataFrame()
    attributes = list(data.head(0))
    attributes.remove(creditability) 

    
    repetitions = 5
    dict_depth = {}
    dict_depth_training = {}
    std_test = []
    std_training = []
    for i in range(100, 1000, 100):
        acc_test = []
        acc_training = []
        for r in range(repetitions):    
            #DIVIDO TRAINING Y TESTEO
            df_list = metrics.cross_validation(data, partition,r)
            test = df_list[0]
            training = pd.DataFrame()
            for j in range(1, partition):
                training = pd.concat([training, df_list[j]], axis=0)
            
            #ejecuto random forest
            fathers = randomForest.random_forest(training,attributes,10,None,None,None,None,i,r)

            accuracy_test = accuracy_random_forest(fathers, test)
            accuracy_training = accuracy_random_forest(fathers, training)
            
           
            acc_test.append(accuracy_test)
            acc_training.append(accuracy_training)
           
        
        std_test.append( np.std(acc_test)/np.sqrt(repetitions))
        dict_depth[i] = np.mean(acc_test)
        std_training.append( np.std(acc_training)/np.sqrt(repetitions))
        
        dict_depth_training[i] = np.mean(acc_training)

    fig, ax = plt.subplots()

    # join the points with a line
    x = dict_depth.keys()
    y = dict_depth.values()
    ax.plot(x,y, ".-", color='cyan' )
    plt.errorbar(x,y, yerr = std_test ,capsize=2, elinewidth=0.5, label="Test")
    


    # join the points with a line
    x = dict_depth_training.keys()
    y = dict_depth_training.values()
    ax.plot(x,y,".-", color='pink')


    plt.errorbar(x,y, yerr = std_training, capsize=2, elinewidth=0.5, label="Training")
    plt.xlabel('Max Nodes')
    plt.ylabel('Tree precision')
    plt.legend(loc='upper right')
    plt.show()


def heatmap_matrix(conf_matrix):
    # create heatmap
    
    sns.heatmap(conf_matrix, annot=True)
    # Convert scientific notation to decimal notation
    fmt = ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x))
    plt.title('Confusion Matrix')
    plt.gca().yaxis.set_major_formatter(fmt)
    plt.show()


def main():
    data = pd.read_csv('./resources/german_credit.csv')
    
    attributes = list(data.head(0))
    data = replaces_process_data(data, 'Duration of Credit (month)', 5)
    data = replaces_process_data(data, 'Credit Amount', 4)
    data = replaces_process_data(data, 'Age (years)', 6)

    
    #plot_variables_count(data)

    partition = 5
    df_list = metrics.cross_validation(data, partition)
    test = df_list[0]
    training = pd.DataFrame()
    for j in range(1, partition):
          training = pd.concat([training, df_list[j]], axis=0)

    attributes.remove(creditability) 


    #########################################################
    #execute ID3
    # values_per_atr = {}
    # for atr in attributes : 
    #     values_per_atr[atr] = training[atr].unique()
    # father = id3_algorithm.id3(training,attributes,values_per_atr,None,None, None,None,None,None) #tree of the training
    # resolve_test(test, father)
    #draw_tree.graph_tree(father)
    #print(id3_algorithm.count_nodes(father))
    


    #########################################################
    #execute Random forest
    fathers = randomForest.random_forest(training,attributes,10,None,None,None,None,None)
    dict_predicted = {}
    for index, row in test.iterrows():
       dict_predicted[index] = {item: 0 for item in creditability_values}

    for father in fathers:
       pred = predicted(test, father)

       for index, value in enumerate(pred):
           dict_predicted[index][value] += 1

    expected_result = expected(test)

    resolve_random_forest(dict_predicted,expected_result)


    #plot_max_nodes_precision(data) 

    #plot_gain(data, attributes)

    #plot_gain_number_values(data) 

    #plot_Random_forest_max_nodes(data)
    
    #plot_Random_forest_probability(data)

    #plot_probability_precision(data)
   
if __name__ == "__main__":
    main() 
