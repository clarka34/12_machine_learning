import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from KMeans import KMeans
from scipy.spatial.distance import cdist

def create_clusters(n_clusters, n_pts, cluster_sep=5):
    """
    Creates clusters.

    Args:
        n_clusters (int): The number of clusters to create
        n_pts (int): The number of points to create
        cluster_sep (float): The minimum distance between cluster centroids

    Returns:
        inputData: A numpy array containing the x and y positions of the points
    """

    x = []
    y = []
    center = np.zeros([n_clusters,2])
    for c in range(n_clusters):
        center[c,:] = [np.random.randint(1,1000),np.random.randint(1,1000)]
        if c>0:
            while cdist([center[c,:]], [center[c-1,:]], metric='euclidean') < cluster_sep:
                center[c,:] = [np.random.randint(1,1000),np.random.randint(1,1000)]
        tempX = np.random.normal(loc=center[c,0], scale=5, size=n_pts)
        tempY = np.random.normal(loc=center[c,1], scale=5, size=n_pts)
        x = np.concatenate((x, tempX))
        y = np.concatenate((y, tempY))

    return np.vstack([x,y]).T

n_clusters = 3

data = create_clusters(n_clusters=n_clusters, n_pts=25, cluster_sep=5)
kmeans_clusterer = KMeans(k=n_clusters, init='random')
kmeans_clusterer.fit(data, visualize=True)
