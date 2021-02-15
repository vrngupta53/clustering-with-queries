import numpy as np
import matplotlib.pyplot as plt
import oracle
import listdict

def clusterise(orcl, eta):
    #initialise variables 
    n = orcl.get_num_samples()
    d = orcl.get_num_features()
    k = orcl.get_num_clusters()
    sample_points = orcl.get_sample_points()
    num_queries = 0
    predicted_clustering = np.full(n, -1, 'i')
    cluster_freq = np.zeros(k, 'i')     #stores number of points discovered of each cluster 
    cluster_sum = np.zeros((k, d))      #stores sum of all points discovered for each cluster 
    cluster_rep = np.full(k, -1, 'i')   #stores index of a representative point from each cluster
    random_sample_size = k * eta + 1    
    removed_cluster = [False] * k      
    remaining_points = listdict.ListDict()  #points that have not been assigned a cluster till now
    for i in range(n):
        remaining_points.add_item(i)

    #find k-1 clusters iteratively
    for i in range(k-1):
        #sample points randomly from remaining points and assign their cluster if not already known
        random_sample = remaining_points.choose_random_items(min(random_sample_size, len(remaining_points)))

        for point in random_sample:
            if predicted_clustering[point] == -1:
                cluster_index, queries = find_cluster(orcl, point, cluster_rep)
                predicted_clustering[point] = cluster_index
                num_queries += queries
                cluster_sum[cluster_index] += sample_points[point]
                if cluster_rep[cluster_index] == -1:
                    cluster_rep[cluster_index] = point
                cluster_freq[cluster_index] += 1
                if cluster_freq[cluster_index] >= eta and not removed_cluster[cluster_index]:
                    break 
        
        #find cluster with most discovered points and estimate its center
        max_freq = 0
        max_freq_index = -1
        for index in range(k):
            if cluster_freq[index] > max_freq and not removed_cluster[index]:
                max_freq = cluster_freq[index]
                max_freq_index = index

        if max_freq_index < 0:
            break 

        estimated_center = cluster_sum[max_freq_index]/cluster_freq[max_freq_index]

        #find radius of cluster by sorting and then binary search
        sorted_points = sorted(remaining_points.items, key=lambda p: np.linalg.norm(sample_points[p] - estimated_center))

        boundary_point_index, queries = binary_search(orcl, sorted_points, cluster_rep[max_freq_index])
        num_queries += queries

        #remove points and update their assigned cluster
        for i in range(boundary_point_index+1):
            predicted_clustering[sorted_points[i]] = max_freq_index
            remaining_points.remove_item(sorted_points[i])

        removed_cluster[max_freq_index] = True

    #find the cluster that remains and assign it all remaining points
    cluster_left = -1
    for i in range(k):
        if not removed_cluster[i]:
            cluster_left = i
            break 

    for point in remaining_points.items:
        predicted_clustering[point] = cluster_left

    return predicted_clustering, num_queries

def find_cluster(orcl, point, cluster_rep):
    index = 0
    queries = 0 
    while cluster_rep[index] >= 0:
        queries += 1
        if orcl.are_same_cluster(point, cluster_rep[index]):
            return index, queries
        index += 1
    
    return index, queries

def binary_search(orcl, sorted_points, cluster_rep_point):
    queries = 0 
    high = len(sorted_points)-1
    low = 0
    while low < high: 
        queries += 1
        mid = (high + low + 1)//2
        if orcl.are_same_cluster(sorted_points[mid], cluster_rep_point):
            low = mid
        else:
            high = mid - 1

    return low, queries

def find_cost(orcl, predicted_clustering):
    n = orcl.get_num_samples()
    d = orcl.get_num_features()
    k = orcl.get_num_clusters()
    sample_points = orcl.get_sample_points()

    centers = np.zeros((k, d))
    freq = np.zeros(k)

    for i in range(n):
        freq[predicted_clustering[i]] += 1
        centers[predicted_clustering[i]] += sample_points[i]

    centers /= freq[:,None]     #broadcast freq and divide to get centers

    total_cost = 0
    for i in range(n):
        total_cost += np.square(np.linalg.norm(sample_points[i] - centers[predicted_clustering[i]]))
    
    return total_cost

def main(): 
    np.seterr(divide='ignore', invalid='ignore')        #ignore divide by 0 warnings
    orcl = oracle.Oracle("data4.npz")
    print("loaded:", orcl.get_filename(), "gamma = ", orcl.get_gamma())
    num_iterations = 20
    lower_limit_eta = 1
    step = 5
    upper_limit_eta = min(int(orcl.get_num_samples()/orcl.get_num_clusters()), 50)
    xs = []
    ys = []
    for eta in range(lower_limit_eta,upper_limit_eta,step):
        success = 0
        total_queries = 0
        total_cost = 0
        for i in range(num_iterations):
            predicted_clustering, queries = clusterise(orcl, eta)
            total_queries += queries
            if orcl.check_predicted_clustering(predicted_clustering):
                success += 1
            total_cost += find_cost(orcl, predicted_clustering)
        
        average_queries = total_queries/num_iterations
        average_cost = total_cost/num_iterations
        xs.append(average_queries)
        ys.append(average_cost/orcl.get_cost())
        print("eta = ", eta, ", success rate:", success, "/", num_iterations, "av. cost = ", average_cost, ", av. queries = ", average_queries, ", av.cost/oracle_cost = ", average_cost/orcl.get_cost())

    #Plot the graph of cost relative to underlying clustering vs number of queries taken
    plt.xlabel('num_queries')
    plt.ylabel('predicted_cluster_cost/oracle_cluster_cost')
    plt.ylim([0.99, 1.01])
    plt.plot(xs, ys)
    plt.savefig('graphs/dataset4.png')

if __name__ == "__main__" :
    main()
