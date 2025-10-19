import numpy as np

import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300, init='random'):
        """
        Initialize the KMeans class.

        Args:
            k (int): The number of clusters
            tol (float): The tolerance used to decide when the model is converged
            max_iter (int): The maximum number of iterations used to fit the model
            init (str): The type of initialization to use, either 'random' or 'kmeans++'
        """       
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.init = init


    def init_clusters(self, X):
        if self.init == 'random':
            # randomly initialize the clusters
            init_indices = np.random.choice(len(X), size=self.k, replace=False)
            centroids = X[init_indices]
            return centroids
            # elif self.init == 'kmeans++':
            # initialize the clusters by selecting a random starting point
            # and then maximizing the distance between subsequent cluster
            # start locations

            # init_ind = np.random.choice(len(X), size=1)
            # distances = np.array([np.linalg.norm(X[init_ind] - X, axis=1) for j in range(self.k)])

            # # Find the furthest point from initial random centroid
            # furthest_centroid = np.argmax(distances, axis=0)

            # for i in range(self.k - 1):
            #     distances = np.array(np.linalg.norm(np.array(centroids) - X, axis=1))

            #     # Find the furthest point from the other centroids
            #     next_centroid_ind = np.argmax(distances, axis=0)
            #     centroids.append(X[next_centroid_ind].tolist())

            # return centroids
        else:
            print("Please enter either 'random' or 'kmeans++' for init")


    def fit(self, X, visualize=False):
        """
        Fits the model on the given X array.

        Args:
            X ((n,m) array): The input data where n is the number of samples
                             and m is the number of features
            init (str) : Type of initialization method, 'random' or 'kmeans++
            visualize (boolean): Boolean to determine whether or not to 
                                 visualize the clusters

        Returns:
            The k centroids from the fitted model.
        """

        # Initialize centroids
        # First centroid is initialized randomly
        # init_indices = np.random.choice(len(X), size=self.k, replace=False)

        num_samples, num_features = X.shape

        # self.centroids = X[init_indices]
        self.centroids = self.init_clusters(X)

        # Loop until you reach the specified maximum number of iterations (max_iter)
        # Break out of the loop before max_iter if the tolerance is reached

        for i in range(self.max_iter):
            # Compute distances between each centroid and all the data points
            distances = np.array([np.linalg.norm(self.centroids[j] - X, axis=1) for j in range(self.k)])

            # For each data point, find the closest centroid
            closest_centroid = np.argmin(distances, axis=0)

            # Update centroids
            # For each centroid, define the new centroid as the center of each new cluster

            new_centroids = np.zeros((self.k, num_features))

            for j in range(self.k):
                cluster = X[closest_centroid == j]
                new_centroids[j] = np.mean(cluster, axis=0)

            # Compute the distance between the previous centroids and the new ones
            # If this distance is lower than our tolerance, stop the algorithm
            converged = []
            for j in range(self.k):
                is_converged = np.mean((self.centroids[j] - new_centroids[j]) / self.centroids[j]) < self.tol
                converged.append(is_converged)
            
            self.centroids = new_centroids
            
            if (np.all(converged)) and (visualize == True):
                fig, ax = plt.subplots(figsize=(8,6))
                # Make a scatter plot of the points, colored by clusters
                ax.scatter(X[:,0], X[:,1],
                           c=closest_centroid,
                           alpha=0.5)
                # Add the centroids to the scatterplot
                ax.scatter(self.centroids[:,0],
                           self.centroids[:,1],
                           marker='X',
                           s=50)
                plt.show()
                break
            elif (np.all(converged)) and (visualize == False):
                break
            

    def predict(self, X):

        return classification