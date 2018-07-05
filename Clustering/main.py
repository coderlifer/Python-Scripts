"""

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from sklearn import preprocessing


class Clustering:
    def __init__(self):
        pass

    def KMeans(self, dataSet, n_clusters):
        kMeans = cluster.KMeans(
            n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
            precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')

        # kMeans.fit(dataSet)
        # clusterAssment = kMeans.labels_
        clusterAssment = kMeans.fit_predict(X=dataSet, y=None)

        return kMeans.cluster_centers_, clusterAssment, kMeans.inertia_

    def MiniBatchKMeans(self, dataSet, n_clusters, batch_size=100):
        miniBatchKMeans = cluster.MiniBatchKMeans(
            n_clusters=n_clusters, init='k-means++', max_iter=100, batch_size=batch_size, verbose=0,
            compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None,
            n_init=3, reassignment_ratio=0.01)

        # miniBatchKMeans.fit(dataSet)
        # clusterAssment = miniBatchKMeans.labels_
        clusterAssment = miniBatchKMeans.fit_predict(X=dataSet, y=None)

        return miniBatchKMeans.cluster_centers_, clusterAssment, miniBatchKMeans.inertia_

    def DBSCAN(self, dataSet, eps, min_samples):
        dbscan = cluster.DBSCAN(
            eps=eps, min_samples=min_samples, metric='euclidean', metric_params=None, algorithm='auto',
            leaf_size=30, p=None, n_jobs=1)

        # dbscan.fit(X=dataSet, y=None, sample_weight=None)
        # clusterAssment = dbscan.labels_
        clusterAssment = dbscan.fit_predict(X=dataSet, y=None, sample_weight=None)

        return dbscan.core_sample_indices_, dbscan.components_, clusterAssment

    def SpectralClustering(self, dataSet, n_clusters):
        sc = cluster.SpectralClustering(
            n_clusters=n_clusters, eigen_solver='arpack', random_state=None, n_init=10, gamma=1.0,
            affinity='nearest_neighbors', n_neighbors=10, eigen_tol=0.0,
            assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=1)

        # sc.fit(X=dataSet, y=None)
        # clusterAssment = sc.labels_
        clusterAssment = sc.fit_predict(X=dataSet, y=None)

        return sc.affinity_matrix_, clusterAssment

    def showCluster(self, dataSet, k, centroids, clusterAssment):
        """Visualize clustering result, only available with 2-D data
        Args:
          dataSet:
          k:
          centroids:
          clusterAssment:

        Return:
        """

        numSamples, dim = dataSet.shape
        if dim != 2:
            print("Sorry! I can not draw because the dimension of your data is not 2!")
            return 1

        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        if k > len(mark):
            print("Sorry! Your k is too large! please contact Zouxy")
            return 1

            # draw all samples
        for i in range(numSamples):
            markIndex = int(clusterAssment[i])
            plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

        if centroids is not None:
            mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
            # draw the centroids
            for i in range(k):
                plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

        plt.show()


def load_data(path):
    data = np.genfromtxt(fname=path, dtype=float, comments='#', delimiter='\t',
                         skip_header=0)

    return data


def plot(data):
    plt.figure()

    plt.plot(data[:, 0], data[:, 1], 'bo')

    plt.show()


if __name__ == '__main__':
    data = load_data('./KMeans/data.txt')
    # plot(data)

    # https://www.cnblogs.com/chaosimple/p/4153167.html
    # http://d0evi1.com/sklearn/preprocessing/
    standardScaler = preprocessing.StandardScaler()
    data = standardScaler.fit_transform(data)
    # plot(data)
    print('mean: {}'.format(np.mean(data)))
    print('std: {}\n'.format(np.std(data)))

    method = 'DBSCAN'

    # DBSCAN param
    dbscan_eps = 0.5
    minPts = 4

    # K-Means param
    n_clusters = 4

    clustering = Clustering()

    if method == 'KMeans':
        # KMeans Clustering
        print('KMeans...')
        cluster_centers, clusterAssment, inertia = clustering.KMeans(data, n_clusters)
        print('\ncluster_centers:\n {}'.format(cluster_centers))
        print('\nclusterAssment:\n {}'.format(clusterAssment))
        print('\ninertia:\n {}'.format(inertia))
    elif method == 'MiniBatchKMeans':
        # MiniBatchKMeans Clustering
        print('MiniBatchKMeans...')
        cluster_centers, clusterAssment, inertia = clustering.MiniBatchKMeans(data, n_clusters)
        print('\ncluster_centers:\n {}'.format(cluster_centers))
        print('\nclusterAssment:\n {}'.format(clusterAssment))
        print('\ninertia:\n {}'.format(inertia))
    elif method == 'DBSCAN':
        # DBSCAN
        print('DBSCAN...')
        core_sample_indices, components, clusterAssment = \
            clustering.DBSCAN(data, eps=dbscan_eps, min_samples=minPts)
        print('\ncore_sample_indices:\n {}'.format(core_sample_indices))
        print('\ncomponents:\n {}'.format(components))
        print('\nclusterAssment:\n {}'.format(clusterAssment))
    elif method == 'SC':
        # Spectral Clustering
        print('SC...')
        affinity_matrix, clusterAssment = clustering.SpectralClustering(data, n_clusters)
        print('\naffinity_matrix:\n {}'.format(affinity_matrix))
        print('\nclusterAssment:\n {}'.format(clusterAssment))
    else:
        raise NotImplementedError('Clustering method [%s] is not recognized!' % method)

    # plot
    clustering.showCluster(data, n_clusters, None, clusterAssment)
