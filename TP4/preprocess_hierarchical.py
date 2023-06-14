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
import preprocess 

genre_preprocess = {'Adventure': 1, 'Comedy': 2, 'Action': 3, 'Drama': 4, 'Crime': 5, 'Fantasy': 6,
                        'Science Fiction': 7,
                        'Horror': 8, 'Romance': 9, 'Mystery': 10, 'Thriller': 11, 'Documentary': 12, 'Animation': 13,
                        'Family': 14, 'History': 15, 'War': 16, 'Western': 17, 'Music': 18, 'TV Movie': 19,
                        'Foreign': 20}

def date_to_int(d):
    return str(d)




def filter_points(cluster,k=10, clusters = [], lenght=10):
    if k == 0:
        clusters.append(cluster.points)
        return clusters
    else:
       clusters_childs = cluster.descendants
       if (len(clusters_childs)) > 1:
            filter_points(clusters_childs[0],k-1,clusters,lenght)
            filter_points(clusters_childs[1],k-1,clusters,lenght)
        
    return clusters


def hierarchical_graph(clusters):
    clusters = filter_points(clusters[0])
    filtered_map = {}
    norm_values_to_categories = {-1.27: 0, -0.08: 1, 1.11: 2}
    genre_counts = np.zeros((len(clusters), len(norm_values_to_categories)), dtype=int)
    
    # Populate genre_counts array
  
    for i,points in enumerate(clusters):
        for point in points:
            p = point[1]
            genre_counts[i][norm_values_to_categories[round(p,2)]] += 1

    print(genre_counts)
    # Create the heatmap
    plt.imshow(genre_counts, cmap='winter')

    # Add genre labels
    genre_labels = [norm_values_to_categories[i] for i in sorted(norm_values_to_categories)]
    plt.xticks(np.arange(len(norm_values_to_categories)), genre_labels, rotation=90)
    plt.yticks(np.arange(len(filtered_map)), filtered_map.keys())

    plt.xlabel("Genres")
    plt.ylabel("Clusters")
    plt.colorbar()
    plt.grid(False)
    plt.show()

def ej1_hierarchical():
    data = preprocess.preprocess_csv()
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
