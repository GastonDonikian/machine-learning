import numpy as np


# PRECONDICIONES:
# Todos los puntos son numéricos.
# Si se quieren mejores resultados, la data deberia llegar normalizada/estandarizada de ante mano.

def _get_k_centroids(data, k):
    return data[np.random.choice(range(len(data)), k, replace=False)]


def _calculate_distance(point_1, point_2):
    return np.sqrt(np.sum((point_1 - point_2) ** 2))


def k_means(data, k, iterations=100, threshold=0.1):
    # Me traigo k centroides 'random', o sea, elijo k puntos
    centroids = _get_k_centroids(data, k)

    # Me armo k 'clusters'
    clusters = [[] for _ in range(k)]
    for _ in range(iterations):
        #Repito para todos los puntos
        for point in data:
            # Calculo la distancia entre cada punto para todos los centroide
            distances = [_calculate_distance(point, centroid) for centroid in centroids]
            # np.argmin me devuelve el indice, o sea, el indice del cluster mas cercano
            cluster_index = np.argmin(distances)
            # le agrego ese punto al cluster
            clusters[cluster_index].append(point)

        #me guardo los nuevos clusters
        new_centroids = []
        for cluster in clusters:
            #el centroide esta actualizado a ser el nuevo 'centro' del cluster
            new_centroids.append(np.mean(cluster, axis=0))

        # me fijo que el threshold no sea menor.
        # se puede ver la doble inclusion pero me dio paja
        max_distance = np.max([_calculate_distance(centroids[i], new_centroids[i]) for i in range(k)])
        if max_distance < threshold:
            break

        centroids = new_centroids

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
