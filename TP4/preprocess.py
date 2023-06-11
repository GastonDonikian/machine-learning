import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from algorithms.kohonen_som import predict, kohonen_som
import algorithms.hierarchical_clustering as hc
import numpy as np
import metrics 

def date_to_int(d):
    return str(d)


def preprocess_csv():
    data_frame = pd.read_csv('./resources/movie_data.csv', delimiter=';')
    column_titles = data_frame.columns.tolist()
    data_frame = data_frame.iloc[1:]

    data_frame = data_frame.dropna()
    data_frame.drop('imdb_id', axis=1, inplace=True)
    data_frame.drop('original_title', axis=1, inplace=True)
    data_frame.drop('overview', axis=1, inplace=True)
    data_frame.drop('release_date', axis=1, inplace=True)

    genres = data_frame.pop("genres")
    normalized_df = (data_frame - data_frame.mean()) / data_frame.std()

    for column in data_frame:
        col = data_frame[column]
        data_frame[column] = (col - col.mean()) / col.std()

    data = data_frame.to_numpy()
    return data


def ejercicio_a():
    data_frame = pd.read_csv('./resources/movie_data.csv', delimiter=';')
    column_titles = data_frame.columns.tolist()
    data_frame = data_frame.iloc[1:]
    numeric_columns = data_frame.apply(pd.to_numeric, errors='coerce').notnull().all()
    numeric_data_frame = data_frame[data_frame.columns[numeric_columns]]

    for i, column in enumerate(numeric_data_frame.columns):
        plt.figure()
        plt.boxplot(numeric_data_frame[column])
        plt.title(column_titles[i])
        plt.xlabel('Columns')
        plt.ylabel('Values')
    # Display the boxplots
    plt.show()

def hierarchical(data):
   print("hierarchical")
   clusters = hc.hierarchical_clustering(data)
   print(clusters)

def main():
    data = preprocess_csv()
    cut_length = len(data) // 100
    cut_array = data[:cut_length]
    hierarchical(cut_array)


def ej1():
    data = preprocess_csv()
    partition = 5
    seed = 2000

    df_list = metrics.cross_validation(data, partition,seed)
    #print(df_list)
    test = df_list[0]

    training = df_list[1]
    for j in range(2, partition):
        training = np.concatenate((training, df_list[j]), axis=0)
    #training = training.to_numpy()
    print("test")
    print(len(test))
    print("training")
    print(len(training))
    epochs = 50
    trained_matrix, mean_distances_per_epoch = kohonen_som(training_set=training,
                                 epochs=epochs,
                                 eta=0.1,
                                 vicinity_radius=5)
    predict(example=data[0], trained_matrix=trained_matrix)
    
    epochs_list = np.array(range(epochs))

    plt.plot(epochs_list, mean_distances_per_epoch)
    plt.show()



if __name__ == "__main__":
    main()
