import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pandas.plotting import parallel_coordinates
import seaborn as sns
import itertools
import math

from sklearn.neighbors import NearestNeighbors
import heapq

from TPF.dataset_analisis import univariable_analisis


def average_distance(points):
    distances = []
    if len(points) <= 1:
        print("Len 0")
        return 0
    for pair in itertools.combinations(points, 2):
        distance = math.dist(pair[0], pair[1])
        distances.append(distance)
    # print("Points")
    # print(points)
    # print("Distances")
    # print(distances)
    return sum(distances) / len(distances)


def analyze(x,eps = 0.3,min_samples=4):
    #function fit dbscan
    print("Started Analyzing")
    # cluster the data into five clusters
    dbscan = DBSCAN(eps = eps, min_samples = min_samples).fit(x) # fitting the model
    labels = dbscan.labels_ # getting the labels
    #print("DBSCAN Result")
    #print(labels)

    column_names = x.columns.tolist()
    # print(column_names)

    clusters = dbscan.fit_predict(x)

    # Print the clusters
    #print("Data Points:\n", x)
    #print("Cluster Labels:\n", clusters)
    n_clusters_ = len(set(labels))
    # print("Labels")
    # print(labels)
    print("Nr of clusters")
    print (n_clusters_)

    #clusters = [x[labels == i] for i in range(-1,n_clusters_)]
    avg_distances = []
    for i in range(n_clusters_-1):
        average = average_distance(x[labels == i].to_numpy())
        if average != 0:
            avg_distances.append(average)
    # print("Avg distances per cluster")
    # print(avg_distances)

    total_avg_distance = sum(avg_distances)/len(avg_distances)
    total_avg_distance = max(avg_distances)
    print(total_avg_distance)
        
    #return total_avg_distance,avg_distances

    # print("Clusters")
    # print(clusters)
    
    # if 1 >0:
    #     return



    for column in x.columns:
        plt.figure()
        plt.title(f'Heatmap of {column} by Cluster')
        
        # Create a copy of the dataframe and add the cluster labels
        ordered_df = x.copy()
        ordered_df['Cluster Labels'] = clusters
        
        # Sort the dataframe by the variable column within each cluster
        ordered_df = ordered_df.sort_values(by=[column])
        sns.heatmap(ordered_df[[column]], cmap='plasma', cbar=False)
        #plt.xlabel(clusters)
        y_labels = [round(value, 2) for value in ordered_df[column]]
        print(y_labels)
        #plt.ylabel(y_labels)
        plt.yticks(ticks=range(len(ordered_df)), labels=y_labels)
        plt.show()
    # Plot the clusters
    #plt.scatter(x[:, 0], x[:,1], c = labels, cmap= "plasma") # plotting the clusters
    #plt.scatter(x[column_names[0]], x[column_names[1]], c = labels, cmap= "plasma") # plotting the clusters
    #plt.xlabel("Avg Sleep Mins") # X-axis label
    #plt.ylabel("Avg Daily Steps") # Y-axis label
    #plt.show() # showing the plot

def run_dbscan(x,eps,min_samples):
    normalized_x=(x-x.mean())/x.std()
    return analyze(normalized_x,eps,min_samples)

# def preprocess_tables(eps,min_samples):
def preprocess_tables():
    daily_sleep_mins_per_user_id = pd.read_csv('./resources/sleepDay_merged.csv', delimiter=',')
    daily_sleep_mins_per_user_id = daily_sleep_mins_per_user_id.dropna()

    daily_sleep_mins_per_user_id = daily_sleep_mins_per_user_id.rename(columns={'SleepDay': 'Date'})
    daily_sleep_mins_per_user_id['Date'] = daily_sleep_mins_per_user_id['Date'].apply(lambda x: x.split(" ")[0])

    daily_activity_per_user_id = pd.read_csv('./resources/dailyActivity_merged.csv', delimiter=',')
    daily_activity_per_user_id = daily_activity_per_user_id.dropna()

    daily_activity_per_user_id = daily_activity_per_user_id.rename(columns={'ActivityDate': 'Date'})
    
    print("Finished Preprocessing")
    
    #x = avg_sleep_mins_per_user_id.set_index('Id').join(avg_daily_steps_per_user_id.set_index('Id')).join(avg_daily_distance_per_user_id.set_index('Id'))
    x = pd.merge(daily_sleep_mins_per_user_id,daily_activity_per_user_id , on=['Id', 'Date'], how='inner')
    print(x.columns)
    x['TotalMinutesAwake'] = x['TotalTimeInBed'] - x['TotalMinutesAsleep']
    x = x[['Calories', 'TotalSteps', 'TotalDistance', 'TotalMinutesAsleep', 'TotalMinutesAwake', 'SedentaryMinutes']]
    # print(x)
    # print(x.size)
    univariable_analisis(x)
    normalized_x=(x-x.mean())/x.std()
    eps = 0.5
    min_samples = 3

    return analyze(normalized_x,eps,min_samples)

    #print(normalized_x)

    points = normalized_x.to_numpy()
    #print(points)
    neighbors_list = [1,2,3,4,5,6,7]
    n_distances = []
    x_values_original = [range(len(points))]
    x_list = []
    for n_neighbors in neighbors_list:
        distances = []
        x_values = x_values_original.copy()
        print("min_neighbors = " + str(n_neighbors))
        for idx,point in enumerate(points):
            smallest_distances = find_closest_points(points, point, n_neighbors)
            #print("Smallest")
            #print(smallest_distances)
            if len(smallest_distances) > 0:
                distances.append(sum(smallest_distances)/len(smallest_distances))
            else:
                del x_values[idx]
        distances.sort()
        n_distances.append(distances)
        x_list.append(x_values)
    
    for i in range(len(n_distances)):
        # plt.plot(np.transpose(x_list[i]), n_distances[i], label =('n'+str(neighbors_list[i])))
        plt.plot(np.transpose(x_values_original[:len(n_distances[i])]), n_distances[i], label ='min_neighbors='+str(neighbors_list[i]))

    plt.xlabel('Points')
    plt.ylabel('Average nearest neighbor distance')
    plt.title('DBSCAN - Método del Codo')
    plt.legend()
    plt.show()

    
    ## Codigo Viejo ##
    # neighbors_list = [1,2,3,4,5,6,7]
    # eps_results = []
    # distance_results = []
    # for n_neighbors in neighbors_list:
    #     eps_values,avg_distance_per_eps = metodo_codo(normalized_x,n_neighbors=n_neighbors)
    #     eps_results.append(eps_values)
    #     distance_results.append(avg_distance_per_eps)
    
    # for i in len(neighbors_list):
    #     plt.plot(eps_results[i], distance_results[i], label =('n'+str(neighbors_list[i])))

    # plt.xlabel('eps')
    # plt.ylabel('Average nearest neighbor distance')
    # plt.title('DBSCAN - Método del Codo ')
    # plt.show()
    ###################

    #return metodo_codo(normalized_x,n_neighbors=4)
    return analyze(normalized_x,eps,min_samples)


def sample_list_vs_maxima_distancia_media_cluster():

    # x = pd.read_csv('./resources/cluster_blobs.csv', delimiter=',')
    # x = x.dropna()


    total_distances = []
    eps = 0.1
    min_samples = 4
    min_sample_list = [1,2,3,4,5,6,7,8,9,10,20,40,60]
    for n in min_sample_list:
        print("Case min_sample_list= " + str(n))
        min_samples = n
        #distance, avg_distances = preprocess_tables(eps,min_samples)
        distance, avg_distances = run_dbscan(x,eps,min_samples)
        total_distances.append(distance)
    plt.scatter(min_sample_list, total_distances)
    plt.title("Points")
    plt.xlabel("Min_Sample")
    plt.ylabel("Maxima Distancia Media")
    plt.show()



# def find_closest_points(points, target_point, n):
#     distances = []
#     for point in points:
#         distance = math.dist(point, target_point)
#         print("Poinyt")
#         print(point)
#         #print(distance)
#         if distance != 0:
#             print(distance)
#             heapq.heappush(distances, (distance, point))
#             #heapq.heappush(distances, (distance, point))
        
#         if len(distances) > n:
#             heapq.heappop(distances)
    
#     closest_points = [d for d, point in distances]
#     l = len(closest_points)
#     #if (l != n):
#         #print("Len= " + str(len(closest_points)))
#     return closest_points

def find_closest_points(points, target_point, n):
    distances = []
    it = 0
    for point in points:
        distance = math.dist(point, target_point) 
        #print("DIstance")
        #print(distance)
        if distance != 0:
            #distances.append((distance, point))
            distances.append(distance)
        
    #print("-----")
    #print(distances)
    distances.sort()
    
    closest_points = distances[:n]
    return closest_points


def display_clustering(data,dbscan):
    labels = dbscan.fit_predict(data)
    plt.figure(figsize = (8,4), dpi = 100)
    sns.scatterplot(data=data,x='X1',y='X2',hue=labels,palette='Set1')
    plt.show()

def metodo_codo(X,n_neighbors=4):
    # data = pd.read_csv('./resources/cluster_blobs.csv', delimiter=',')
    # X = data.dropna()
    X_original = X

    #X_original=(X-X.mean())/X.std()
    #print(X)
    #X = X.dropna()

    # Definir rango de valores de eps
    #eps_values = np.linspace(0.1, 2.0, num=20)
    eps_values = np.linspace(0.12, 1.5, num=20)

    # Calcular la distancia media del vecino más cercano para cada eps
    #avg_distances = []
    avg_distance_per_eps = []
    for eps in eps_values:
        X = X_original.copy()
        print("eps= "+str(eps))
        dbscan = DBSCAN(eps=eps,min_samples=n_neighbors).fit(X)
        dbscan.fit(X)
        
        # Calcular la distancia media del vecino más cercano
        # distances = NearestNeighbors(n_neighbors=2).fit(X).kneighbors()[0][:, 1]
        # avg_distances.append(np.mean(distances))

        
        labels = dbscan.labels_
        n_clusters_ = len(set(labels))
        
        distances_per_cluster = []
        for i in range(n_clusters_-1):
            cluster = X[labels == i].to_numpy()
            distances_per_points = []
            for point in cluster:
                smallest_distances = find_closest_points(cluster, point, n_neighbors)
                #print("Smallest")
                #print(smallest_distances)
                if len(smallest_distances) >0:
                    distances_per_points.append(sum(smallest_distances)/len(smallest_distances))
                # print("Smallest")
                # print(smallest_distances)
                # return
                
            distances_per_cluster.append(sum(distances_per_points)/len(distances_per_points))
        if len(distances_per_cluster) > 0:
            avg_distance_per_eps.append(sum(distances_per_cluster)/len(distances_per_cluster))
            print("Avg distance= " + str(sum(distances_per_cluster)/len(distances_per_cluster)))
        else:
            print(" Dio 0")
        #display_clustering(X_original,dbscan)
    
    # Graficar clustering
    
    return eps_values, avg_distance_per_eps

    # Graficar los resultados
    print("eps_values")
    print(eps_values)
    print("avg_distance_per_eps")
    print(avg_distance_per_eps)
    plt.plot(eps_values, avg_distance_per_eps, marker='o')
    plt.xlabel('eps')
    plt.ylabel('Average nearest neighbor distance')
    plt.title('DBSCAN - Método del Codo - min_neighbors=' + str(n_neighbors))
    plt.show()



if __name__ == "__main__":
    preprocess_tables()
    #metodo_codo()
