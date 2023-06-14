import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from algorithms.kohonen_som import predict, kohonen_som
import algorithms.hierarchical_clustering as hc
from algorithms.k_medias import k_means, get_cluster_by_point
import numpy as np
import metrics
import pickle


def date_to_int(d):
    return str(d)


def grouping_csv():
    data_frame = pd.read_csv('./resources/movie_data.csv', delimiter=';')
    column_titles = data_frame.columns.tolist()
    data_frame = data_frame.iloc[1:]
    data_frame = data_frame.dropna()
    data_frame.drop('imdb_id', axis=1, inplace=True)
    data_frame.drop('original_title', axis=1, inplace=True)
    data_frame.drop('overview', axis=1, inplace=True)
    data_frame.drop('release_date', axis=1, inplace=True)
    variables = ['budget', 'popularity', 'production_companies', 'production_countries', 'revenue', 'runtime', 'spoken_languages', 'vote_average', 'vote_count']
    genre_preprocess = ['Adventure', 'Comedy', 'Action', 'Drama', 'Crime', 'Fantasy','Science Fiction',
                        'Horror', 'Romance', 'Mystery', 'Thriller', 'Documentary', 'Animation',
                        'Family', 'History', 'War', 'Western', 'Music', 'TV Movie','Foreign']
    for variable in variables:
        bars = data_frame.groupby(['genres'])[variable].mean()
        err = data_frame.groupby(['genres'])[variable].std()
        plt.barh(genre_preprocess, bars, xerr=err, align='center')
        plt.xlabel('Average: ' + variable)
        plt.ylabel('Values')
        plt.savefig('./images/genero_vs_average/' + variable + '.png')
        plt.show()



def preprocess_csv():
    data_frame = pd.read_csv('./resources/movie_data.csv', delimiter=';')
    column_titles = data_frame.columns.tolist()
    data_frame = data_frame.iloc[1:]
    genre_preprocess = {'Adventure': 1, 'Comedy': 2, 'Action': 3, 'Drama': 4, 'Crime': 5, 'Fantasy': 6,
                        'Science Fiction': 7,
                        'Horror': 8, 'Romance': 9, 'Mystery': 10, 'Thriller': 11, 'Documentary': 12, 'Animation': 13,
                        'Family': 14, 'History': 15, 'War': 16, 'Western': 17, 'Music': 18, 'TV Movie': 19,
                        'Foreign': 20}
    
    valid_categories = ['Action','Comedy','Drama']
    data_frame = data_frame[data_frame['genres'].isin(valid_categories)] 
    
    for k in genre_preprocess:
        data_frame['genres'] = data_frame['genres'].replace([k], [int(genre_preprocess[k])])
    data_frame = data_frame.dropna()
    data_frame.drop('imdb_id', axis=1, inplace=True)
    data_frame.drop('original_title', axis=1, inplace=True)
    data_frame.drop('overview', axis=1, inplace=True)
    data_frame.drop('release_date', axis=1, inplace=True)
    print("Old genres")
    print(data_frame['genres'])
    for column in data_frame:
        col = data_frame[column]
        data_frame[column] = (col - col.mean()) / col.std()
    print("Normalized genres")
    print(data_frame['genres'])
    #print(data_frame)
    data = data_frame.to_numpy()
    return data


def ejercicio_a():
    grouping_csv()
    data_frame = preprocess_csv()
    column_titles = data_frame.columns.tolist()
    data_frame = data_frame.iloc[1:]
    plt.rcParams["figure.figsize"] = [17.50, 7.50]
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    plt.boxplot(data_frame, labels=column_titles)
    plt.xlabel('Columns')
    plt.grid()
    plt.ylabel('Values')
    # plt.savefig('./images/analisis_univariado/boxplot_variables_no_estandarizadas.png')
    plt.show()


def hierarchical_graph(clusters):
    # print("HOLA")
    print(clusters)


def ej1_hierarchical():
    data = preprocess_csv()
    cut_length = len(data) // 400

    cut_array = data[:cut_length]
    print(len(cut_array))
    print("in")
    clusters = hc.hierarchical_clustering(cut_array)
    print("out")
    hierarchical_graph(clusters)


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
    for ki in [3]:
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

    categories = np.zeros((k,k,3))
    for t in test:
        category = t[1]
        i,j,min_distance = predict(example=t, trained_matrix=trained_matrix, popularity_matrix=popularity_matrix)
        categories[i][j] += 1
    for t in test:
        category = t[1]
    print(categories)

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
    #print(len(test))
    print("training")
    #print(len(training))

    k_media_codo(training)
    k = 7
    centroids, clusters = k_means(training, k, iterations=2000, threshold=0.001)
    # print("Centroids")
    # print(centroids)
    # print("Clusters")
    # print(clusters)
    norm_values_to_categories = {'-1.273803': 2, '-0.081453': 3, '1.110897': 4}
    cats = []
    for i in range(0,k):
        cats.append({})
        #print("Len cluster")
        #print(len(clusters[i]))
        for x in clusters[i]:
            if str(round(x[1],6)) not in cats[i]:
                cats[i][str(round(x[1],6))] = 1
            else:
                cats[i][str(round(x[1],6))] = cats[i][str(round(x[1],6))] + 1
    print(cats)
    predominant_category = []
    for i in range(0,k):
        predominant_category.append(norm_values_to_categories[max(cats[i], key=cats[i].get)])
    print("Predominant category by cluster")
    print(predominant_category)

    mat = np.zeros((3,3))

    for t in test:
        genre = norm_values_to_categories[str(round(t[1],6))]
        categ = predominant_category[get_cluster_by_point(point=t,centroids=centroids)]
        mat[genre-2][categ-2] = mat[genre-2][categ-2] + 1
    print("Matrix")
    print(mat)
    heatmap = plt.pcolor(mat)
    plt.colorbar()
    plt.show()

def k_media_codo(training):
    distances_avg = []
    k = range(2,16)

    for i in range(2,16):
        
        centroids, clusters = k_means(training, i, iterations=2000, threshold=0.001)
        distance = 0
        distances = []
        for i,cluster in enumerate(clusters):
            distance = np.linalg.norm(centroids[i] - cluster, axis=1)
            distance = (sum(distance))/len(distance)
            distances.append(distance)
        distances_avg.append(sum(distances)/len(distances))

    plt.title("Metodo del Codo")
    plt.xlabel('k')
    plt.ylabel('Distancia Media')
    plt.plot(k,distances_avg, 'o-')
    #plt.gca().legend(('k=3','k=5','k=7','k=9'))
    plt.show()            



if __name__ == "__main__":
    # main()
    #ej1_kohonen()
    ej1_k_medias()
    #ej1_hierarchical()
    #ej1_k_medias()

