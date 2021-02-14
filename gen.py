import numpy as np
import math
from sklearn.datasets import make_blobs

#Checks if input cluster satisfies voronoi partition and calculates gamma value
def check_validity(X, y, n_clusters):
    #calculate centers of each cluster
    n_samples, n_features = X.shape
    n_centers = n_clusters

    centers = np.zeros((n_centers, n_features))
    freq = np.zeros(n_centers)

    for i in range(n_samples):
        freq[y[i]] += 1
        centers[y[i]] += X[i]

    centers /= freq[:,None]     #broadcast freq and divide to get centers

    #check if clustering satisfies voronoi partition
    is_voronoi_partition = True 
    for i in range(n_samples):
        min_dist = math.inf
        min_dist_center_index = -1

        for j in range(n_centers):
            dist_from_center = np.linalg.norm(X[i] - centers[j])
            if(dist_from_center < min_dist):
                min_dist = dist_from_center
                min_dist_center_index = j
        
        if(min_dist_center_index != y[i]):
            is_voronoi_partition = False
            break

    #calculate gamma by checking all distances of points from centers
    gamma = math.inf
    for i in range(n_centers):
        min_dist_from_center = math.inf
        max_dist_from_center = 0
        for j in range(n_samples):
            sample_dist = np.linalg.norm(centers[i] - X[j])
            if(y[j] == i): 
                max_dist_from_center = max(max_dist_from_center, sample_dist)
            else:
                min_dist_from_center = min(min_dist_from_center, sample_dist)
        gamma = min(gamma, min_dist_from_center/max_dist_from_center)

    #calculate cost of the given clustering
    total_cost = 0
    for i in range(n_samples):
        total_cost += np.square(np.linalg.norm(X[i] - centers[y[i]]))
        
    return gamma, centers, is_voronoi_partition, total_cost

def main():
    n_samples = 50000
    n_clusters = 10
    n_features = 20
    cluster_std = 12.0

    while True:
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=cluster_std, center_box=[-50, 50])
        gamma, centers, is_voronoi_partition, cost = check_validity(X, y, n_clusters)
        print("is_voronoi_partition = ", is_voronoi_partition, "gamma =", gamma)

        if(is_voronoi_partition and 1 < gamma < 1.05): 
            np.savez("dataset/data5", X=X, y=y, k=n_clusters, g=gamma, c=cost)
            break
    
if __name__ == "__main__":
    main()