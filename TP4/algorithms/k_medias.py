import numpy as np
import random

# PRECONDICIONES:
# Todos los puntos son numéricos.
# Si se quieren mejores resultados, la data deberia llegar normalizada/estandarizada de ante mano.

def _get_k_centroids(data, k):
    return data[np.random.choice(range(len(data)), k, replace=False)]


def _calculate_distance(point_1, point_2):
    return np.linalg.norm(np.subtract(point_1, point_2), 2)

def initialize_clusters(data,k):
    clusters = [[] for _ in range(k)]
    for d in data:
        clusters[random.randint(0, k-1)].append(d)
    return clusters

def update_centroid(idx, centroids, clusters):
    if len(clusters[idx]) == 0:
        return clusters[idx]
    s = np.zeros(len(clusters[idx][0]))
    for c in clusters[idx]:
        s = np.sum([s,c], axis=0)
    return (1/len(clusters[idx])) * s

def k_means(data, k, iterations=1000, threshold=0.001):
    # Me traigo k centroides 'random', o sea, elijo k puntos
    centroids = _get_k_centroids(data, k)
    print(centroids)


    # Me armo k 'clusters'
    clusters = initialize_clusters(data,k)
    #print(clusters)
    new_centroids = np.empty((k,len(data[0])))
    #print(new_centroids)
    it = 0
    for _ in range(iterations):
        print(it)
        # print("Clusters:")
        # for clu in clusters:
        #     print(len(clu))
        # Actualizo centroides     
        for idx in range(0,k):
            if len(clusters[idx]) != 0:
                new_centroids[idx] = update_centroid(idx, centroids, clusters)
            else:
                new_centroids[idx] = []

        #Repito para todos los puntos
        for point in data:
            # Calculo la distancia entre cada punto para todos los centroide
            distances = [_calculate_distance(point, centroid) for centroid in centroids]
            # np.argmin me devuelve el indice, o sea, el indice del cluster mas cercano
            cluster_index = np.argmin(distances)
            # le agrego ese punto al cluster
            clusters[cluster_index].append(point)

        

        # me fijo que el threshold no sea menor.
        # se puede ver la doble inclusion pero me dio paja
        max_distance = np.max([_calculate_distance(centroids[i], new_centroids[i]) for i in range(k)])
        print("Max distance")
        print(max_distance)
        if max_distance < threshold:
            break

        centroids = new_centroids
        it += 1

    return centroids, clusters






def ejemplo():
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    k = 2
    centroids, clusters = k_means(data, k)
    print("Final centroids:", centroids)
    print("Clusters:", clusters)
def main():
    ejemplo()


if __name__ == "__main__":
    main()
