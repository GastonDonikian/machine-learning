import numpy as np


def _calculate_distance(point_1, point_2):
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)
    return np.sqrt(np.sum((point_1 - point_2) ** 2))


# Algo asi
# p1 = (1,1)
# p2 = (2,2)
# p3 = (5,5)
#      p1 p2 p3
# p1 ( 0  1  4 )
# p2 ( .  .  . )
# p3 ( .  .  . )

def _get_distance_matrix(points):
    distance_matrix = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance_matrix[i][j] = _calculate_distance(point_1=points[i], point_2=points[j])
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix


def hierarchical_clustering(points, number_of_clusters):
    distance_matrix = _get_distance_matrix(points=points)
    # Cada cluster es un array con un unico elemento, ahora tengo que mergear
    clusters = [[i] for i in range(len(points))]
    if number_of_clusters < 1:
        raise ValueError("Number of clusters can't be less than 1.")
    while len(clusters) > number_of_clusters:
        min_distance = np.inf
        merge_indices = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Mapeo todos los pares de distancias a un array, y hago el minimo
                distance = np.min([distance_matrix[p1, p2] for p1 in clusters[i] for p2 in clusters[j]])
                if distance < min_distance:
                    min_distance = distance
                    # me guardo la tupla de indices
                    merge_indices = (i, j)

        # Merge the closest clusters
        merged_cluster = clusters[merge_indices[0]] + clusters[merge_indices[1]]
        clusters = [c for idx, c in enumerate(clusters) if idx not in merge_indices]
        clusters.append(merged_cluster)

    final_clusters = []
    for cluster in clusters:
        points_in_cluster = [points[i] for i in cluster]
        final_clusters.append(points_in_cluster)

    return final_clusters




def ejemplo():
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    num_clusters = 4
    clusters = hierarchical_clustering(data, num_clusters)
    print("Final clusters:", clusters)
def main():
    ejemplo()


if __name__ == "__main__":
    main()