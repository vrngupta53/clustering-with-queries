import os, random
import numpy as np 

class Oracle:
    def __init__(self, filename=None):
        self.filename = None
        self.X = None
        self.y = None
        self.n_samples = None
        self.n_features = None
        self.n_clusters = None
        self.gamma = None
        self.load_dataset(filename)
        

    def load_dataset(self, filename):
        if filename is None:
            filename = random.choice(os.listdir("dataset"))
        self.filename = filename
        data = np.load(os.path.join("dataset", filename))
        self.X = data['X']
        self.y = data['y']
        self.n_clusters = int(data['k'])
        self.gamma = float(data['g'])
        self.n_samples, self.n_features = self.X.shape

    def get_filename(self):
        return self.filename

    def get_num_samples(self):
        return self.n_samples

    def get_num_features(self):
        return self.n_features

    def get_num_clusters(self):
        return self.n_clusters

    def get_gamma(self):
        return self.gamma

    def get_sample_points(self):
        return self.X

    def are_same_cluster(self, sample1, sample2):
        return self.y[sample1] == self.y[sample2]

    def check_predicted_clustering(self, predicted_clustering):
        if predicted_clustering.shape != self.y.shape:
            raise ValueError("Predicted array has shape", predicted_clustering.shape, ",should be", self.y.shape)
        
        cluster_map = [0]*self.n_clusters
        for i in range(self.n_samples):
            cluster_map[predicted_clustering[i]] = self.y[i]
        
        for i in range(self.n_samples):
            predicted_clustering[i] = cluster_map[predicted_clustering[i]]
        
        return (self.y == predicted_clustering).all()