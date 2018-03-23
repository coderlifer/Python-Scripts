
import numpy as np
# import tensorflow as tf

import matplotlib.pyplot as plt


class KMeans(object):
    def __init__(self, k=4, distance='l2', max_iter=1600, centroids=None, data=None):
        self.k = k
        self.distance = distance
        self.max_iter = max_iter
        self.data = data
        self.centroids = centroids
        self.labels = np.zeros((self.data.shape[0], 1))

        self.num_data = data.shape[0]
        self.num_dim = data.shape[1]

        self.init_centroids()

    def init_centroids(self):
        """Initialize centroids"""

        if self.centroids is None:
            # select k items randomly as initial centroids 
            indices = np.random.choice(self.num_data, self.k)
            self.centroids = self.data[indices]

    def l2_dist(self, data):
        """Euclid distance"""

        # dists[i, j]: distance between centroids[i] and data[j]
        # dists = np.zeros((self.k, self.num_dim))

        # use matrix multiplication to accelerate the algorithm
        data_square = np.sum(np.square(data), axis=1)  # m * 1
        data_square_expand = np.tile(data_square, (1, self.k))  # m * k

        centroids_square = np.sum(np.square(self.centroids), axis=1)  # k * 1
        centroids_square_expand = np.transpose(np.tile(centroids_square, (1, self.num_data)))  # m * k

        multiply_rs = np.dot(data, self.centroids.T)  # m * k

        square_rs = np.abs(np.subtract(np.add(data_square_expand, centroids_square_expand), multiply_rs * 2))

        dists = np.sqrt(square_rs)  # k * n

        return dists  # n * k

    def manhattan_dist(self, data):
        """Manhattan distance"""

        dists = np.zeros((self.k, self.num_dim))

        centroids_repeat = np.repeat(self.centroids, self.num_data, axis=0)
        centroids_repeat = np.reshape(centroids_repeat, (self.k, self.num_data, self.num_dim))
        dists = np.abs(np.sum(centroids_repeat - data, axis=2))  # k * n

        return np.transpose(dists)  # n * k

    def cal_distance(self, data):
        """Calculate distance"""

        dists = None
        if self.distance == 'l2':
            dists = self.l2_dist(data)
        elif self.distance == 'manhattan_dist':
            dists = self.manhattan_dist(data)

        return dists

    def update_centroids(self, dists):
        """Update centroids and labels"""

        centroids_old = self.centroids

        # update labels, remember to reshape
        print("update_centroids, dists.shape: {0}".format(dists.shape))  # n * k
        self.labels = np.reshape(np.argsort(dists, axis=1)[:, 0], (-1, 1))  # n * 1

        # update centroids
        labels_flatten = self.labels.reshape((-1, 1))
        for i in range(self.k):
            # get data items whose label is i
            label_idx = np.where(labels_flatten == i)

            label_idx = np.asarray(label_idx[0])
            data_i = self.data[label_idx]

            # calculate new centroids by mean value
            mean_val = np.mean(data_i, axis=0)
            new_centroids = np.reshape(mean_val, (-1, self.num_dim))
            self.centroids[i] = new_centroids

        # if centroids changed or not
        changed = np.sum(centroids_old - self.centroids)
        if changed == 0:
            is_changed = False
        else:
            is_changed = True

        return is_changed

    def fit(self, data):
        """Train process"""

        if not isinstance(data, np.ndarray) or isinstance(data, np.matrixlib.defmatrix.matrix):
            try:
                data = np.asarray(data)
            except:
                raise TypeError("numpy.ndarray resuired for data")

        for _ in range(self.max_iter):
            dists = self.cal_distance(self.data)
            is_changed = self.update_centroids(dists)
            if not is_changed:
                break

        return [self.centroids, self.labels]

    def predict(self, data):
        """Predict"""

        dists = self.cal_distance(data)
        labels = np.reshape(np.argsort(dists, axis=1)[:, 0], (-1, 1))

        return labels

    def showCluster(self, dataSet, k, centroids, clusterAssment):
        """Visualize clustering result, only available with 2-D data"""

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
            markIndex = int(clusterAssment[i, 0])
            plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # draw the centroids  
        for i in range(k):
            plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

        plt.show()


if __name__ == "__main__":
    # step 1: reading data...
    print("step 1: load data...")
    dataSet = []
    fileIn = open('./data.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
    dataSet = np.mat(dataSet)

    # step 2: clustering...
    print("step 2: clustering...")
    kmeans = KMeans(k=4, data=dataSet)
    centroids, clusterAssment = kmeans.fit(dataSet)

    # step 3: show the result
    print("step 3: show the result...")
    kmeans.showCluster(dataSet, 4, centroids, clusterAssment)
