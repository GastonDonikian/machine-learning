import numpy as np



class Cluster:
    def __init__(self,points,indices):
        self.points = points
        self.indices = indices
        self.centroid =  np.mean(points, axis=0)

    def calculate_distance(self,centroids):
            return  np.linalg.norm(self.centroid - centroids, axis=1)
            
        
    def __str__(self):
        return str(self.points)
        
    def __repr__(self) -> str:
        return self.__str__()



# Algo asi
# p1 = (1,1)
# p2 = (2,2)
# p3 = (5,5)
#      p1 p2 p3
# p1 ( 0  1  4 )
# p2 ( .  .  . )
# p3 ( .  .  . )

def _get_distance_matrix(points):
    print("_get_distance_matrix")
    #points = np.array(points)
    distance_matrix = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        distance_matrix[i, :] = np.sqrt(np.sum((points[i] - points) ** 2, axis=1))
    np.fill_diagonal(distance_matrix, np.inf)
    return distance_matrix


def _calculate_distance_clusters_min_distance(distance_matrix, cluster1, cluster2):
    indices1 = [p1[1] for p1 in cluster1]
    indices2 = [p2[1] for p2 in cluster2]
    distance = np.min(np.take(distance_matrix, indices1, axis=0, mode='raise').take(indices2, axis=1, mode='raise'))
    return distance


def hierarchical_clustering(points):
    threshold = 0.0001
    distance_matrix = _get_distance_matrix(points=points)
    final_clusters = []
    clusters = []
    centroids = []
    # Cada cluster es un array con un unico elemento, ahora tengo que mergear
    for i in range(len(points)):
       clusters.append(Cluster([points[i]],[i]))
    clusters = np.array(clusters, dtype=object)    
    centroids =  np.array([cluster.centroid for cluster in clusters])

    while len(clusters) > 1:
        print(len(clusters),'          ', end='\r')
        merge_indices = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
        merged_cluster = Cluster(
            clusters[merge_indices[0]].points + clusters[merge_indices[1]].points,
            clusters[merge_indices[0]].indices + clusters[merge_indices[1]].indices
        )
        #clusters = [c for idx, c in enumerate(clusters) if idx not in merge_indices]
        clusters = np.delete(clusters,merge_indices[1])
        distance_matrix = np.delete(distance_matrix,merge_indices[1],0)
        distance_matrix = np.delete(distance_matrix,merge_indices[1],1)
        centroids = np.delete(centroids,merge_indices[1],0)

        centroids[merge_indices[0]] = merged_cluster.centroid
        clusters[merge_indices[0]] = merged_cluster
        distances = merged_cluster.calculate_distance(centroids)
        final_clusters.append(merged_cluster)
        distance_matrix[merge_indices[0]] = distances
        distance_matrix[:, merge_indices[0]] = distances
        distance_matrix[merge_indices[0], merge_indices[0]] = np.inf
    #print(final_clusters)  
    return final_clusters

#calculate_distance_clusters = { 'min': _calculate_distance_clusters_min_distance,'centroid':_calulate_centroid_distance}


def ejemplo():
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    clusters = hierarchical_clustering(data)
    print("Final clusters:", clusters)
def main():
    ejemplo()


if __name__ == "__main__":
    main()