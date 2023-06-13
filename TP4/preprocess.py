import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from algorithms.kohonen_som import predict, kohonen_som
import algorithms.hierarchical_clustering as hc
from algorithms.k_medias import k_means
import numpy as np
import metrics
import pickle
import copy



def date_to_int(d):
    return str(d)


def preprocess_csv():
    data_frame = pd.read_csv('./resources/movie_data.csv', delimiter=';')
    column_titles = data_frame.columns.tolist()
    data_frame = data_frame.iloc[1:]
    genre_preprocess = {'Adventure': 1, 'Comedy': 2, 'Action': 3, 'Drama': 4, 'Crime': 5, 'Fantasy': 6,
                        'Science Fiction': 7,
                        'Horror': 8, 'Romance': 9, 'Mystery': 10, 'Thriller': 11, 'Documentary': 12, 'Animation': 13,
                        'Family': 14, 'History': 15, 'War': 16, 'Western': 17, 'Music': 18, 'TV Movie': 19,
                        'Foreign': 20}
    for k in genre_preprocess:
        data_frame['genres'] = data_frame['genres'].replace([k], [int(genre_preprocess[k])])
    data_frame = data_frame.dropna()
    data_frame.drop('imdb_id', axis=1, inplace=True)
    data_frame.drop('original_title', axis=1, inplace=True)
    data_frame.drop('overview', axis=1, inplace=True)
    data_frame.drop('release_date', axis=1, inplace=True)

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


def save_var(var):
    file = open('Python.txt', 'w')
    pickle.dump(var, file)
    file.close()


def retrieve_var():
    with open('Python.txt', 'rb') as f:
        return pickle.load(f)


def ej1_kohonen():
    data = preprocess_csv()
    partition = 5
    seed = 2000

    df_list = metrics.cross_validation(data, partition, seed)
    # print(df_list)
    test = df_list[0]

    training = df_list[1]
    for j in range(2, partition):
        training = np.concatenate((training, df_list[j]), axis=0)
    # training = training.to_numpy()
    print("test")
    print(len(test))
    print("training")
    print(len(training))

    mean_distances_by_k = []
    epochs = 100
    epochs_list = np.array(range(epochs))
    for ki in [7]:
        # k=7
        k = ki
        rows = k
        cols = k
        trained_matrix, mean_distances_per_epoch, popularity_matrix = kohonen_som(training_set=training,
                                                                                  epochs=epochs,
                                                                                  eta=0.1,
                                                                                  vicinity_radius=5, rows=rows,
                                                                                  cols=cols)
        mean_distances_by_k.append(mean_distances_per_epoch)
        # plt.plot(epochs_list, mean_distances_per_epoch, label="k=" +str(k))

    # predict(example=data[0], trained_matrix=trained_matrix, popularity_matrix=popularity_matrix)
    # save_var(mean_distances_by_k)

    # print("Popularity matrix")
    # print(popularity_matrix)
    plt.title("Distancia promedio por epoca")
    plt.xlabel('Epocas')
    plt.ylabel('Distancia Media')
    for line in mean_distances_by_k:
        plt.plot(epochs_list, line)
    # plt.gca().legend(('k=3','k=5','k=7','k=9'))
    plt.show()

    plt.title("Popularity Matrix Heat Map")
    # plt.imshow(popularity_matrix, cmap='hot', interpolation='nearest')
    heatmap = plt.pcolor(popularity_matrix)
    plt.colorbar()
    plt.show()


def ej1_k_medias():
    data = preprocess_csv()
    partition = 5
    seed = 2000

    df_list = metrics.cross_validation(data, partition, seed)
    # print(df_list)
    test = df_list[0]

    training = df_list[1]
    for j in range(2, partition):
        training = np.concatenate((training, df_list[j]), axis=0)
    # training = training.to_numpy()
    print("test")
    print(len(test))
    print("training")
    print(len(training))
    k = 7
    centroids, clusters = k_means(training, k, iterations=2000, threshold=0.001)
    # print("Centroids")
    # print(centroids)
    # print("Clusters")
    # print(clusters)


if __name__ == "__main__":
    #main()
    #ej1_kohonen()
    ej1_k_medias()
    #ej1_k_medias()
