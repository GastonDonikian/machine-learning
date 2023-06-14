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

genre_preprocess = {'Adventure': 1, 'Comedy': 2, 'Action': 3, 'Drama': 4, 'Crime': 5, 'Fantasy': 6,
                        'Science Fiction': 7,
                        'Horror': 8, 'Romance': 9, 'Mystery': 10, 'Thriller': 11, 'Documentary': 12, 'Animation': 13,
                        'Family': 14, 'History': 15, 'War': 16, 'Western': 17, 'Music': 18, 'TV Movie': 19,
                        'Foreign': 20}

def date_to_int(d):
    return str(d)


def preprocess_csv():
    data_frame = pd.read_csv('./resources/movie_data.csv', delimiter=';')
    column_titles = data_frame.columns.tolist()
    data_frame = data_frame.iloc[1:]

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
    print(data[0])
    return data


def filter_points(cluster,k=10, map_clusters={}, lenght=10):
    if k == 0:
        map_clusters[lenght] = cluster.points
        return map_clusters
    else:
       clusters_childs = cluster.descendants
       if (len(clusters_childs)) > 1:
            filter_points(clusters_childs[0],k-1,map_clusters,lenght)
            filter_points(clusters_childs[1],k-2,map_clusters,lenght)
       r = k - lenght
       points_filter = []
       for i in range(0,r):
           points_filter.extend(map_clusters[i])
       points = copy.copy(cluster.points)
       map_clusters[r] =  [element for element in points if element not in points_filter]
       return map_clusters

def hierarchical_graph(clusters):
    map_cluster = filter_points(clusters[0])
    filtered_map = {}
 
    genre_counts = np.zeros((len(map_cluster), len(genre_preprocess)), dtype=int)

    # Populate genre_counts array
    for key, points in map_cluster.items():
        for point in points:
            print(point)
            genre_counts[key][point[1]] += 1

    # Create the heatmap
    plt.imshow(genre_counts, cmap='hot')

    # Add genre labels
    genre_labels = [genre_preprocess[i] for i in sorted(genre_preprocess)]
    plt.xticks(np.arange(len(genre_preprocess)), genre_labels, rotation=90)
    plt.yticks(np.arange(len(filtered_map)), filtered_map.keys())

    plt.xlabel("Genres")
    plt.ylabel("Clusters")
    plt.colorbar()
    plt.grid(False)
    plt.show()

def ej1_hierarchical():
    data = preprocess_csv()
    print(len(data))
    cut_length = len(data)//10
    cut_array = data[:cut_length]
    print(len(cut_array))
    print("in")
    clusters = hc.hierarchical_clustering(cut_array)
    print("out")
    hierarchical_graph(clusters)





if __name__ == "__main__":
    #main()
    #ej1_kohonen()
    #ej1_k_medias()
    ej1_hierarchical()
    #ej1_k_medias()
