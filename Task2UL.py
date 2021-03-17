#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:53:57 2020

@author: tobias
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:41:43 2020

@author: tobias
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
## as functions are taken from existing libraries their execution and
## formalism is not described in detail
###!!Some of the techniques are only documented using one of the three clusters
### cluster1, cluster2 and cluster3 to reducy redundancy. Every algorithm
### was executed in the same way only differening by using on of the 3 metioned
### data sets.

#read in data and store as cluster1, cluster2 and cluster 3
cluster1 = np.loadtxt("noCluster2_1K.csv", delimiter=',', skiprows=1)
cluster2 = np.loadtxt("noCluster2_2K.csv", delimiter=',', skiprows=1)
cluster3 = np.loadtxt("noCluster3_1K.csv", delimiter=',', skiprows=1)


#data is centered by fitted and following transformation
cluster1 = StandardScaler().fit_transform(cluster1)
cluster2 = StandardScaler().fit_transform(cluster2)
cluster3 = StandardScaler().fit_transform(cluster3)

#print cluster1
plt.scatter(cluster1[:, 0], cluster1[:, 1], s=50)
plt.xlabel("X")
plt.ylabel("Y")

#print cluster2
plt.scatter(cluster2[:, 0], cluster2[:, 1], s=50)
plt.xlabel("X")
plt.ylabel("Y")

#print cluster3
plt.scatter(cluster3[:, 0], cluster3[:, 1], s=50)
plt.xlabel("X")
plt.ylabel("Y")


## DBSCAN
#Initialize X as used data for DBSCAN
X = cluster1

plt.scatter(X[:, 0], X[:, 1], s=50)

#execute DBSCAN
db = DBSCAN(eps=0.4, min_samples=9).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


###Execute KMeans
from sklearn.cluster import KMeans


X = cluster1


# plot
plt.scatter(
   X[:, 0], X[:, 1],
   c='white', marker='o',
   edgecolor='black', s=50
)
plt.show()

from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=3, init='k-means++',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)
# plot the 3 clusters
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)
#Enable in case of 3 initial centroids, e.g. cluster3
#plt.scatter(
    #X[y_km == 2, 0], X[y_km == 2, 1],
    #s=50, c='lightblue',
    #marker='v', edgecolor='black',
    #label='cluster 3'
#)
# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

)
#Reference used for KMeans https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

##Execute hierarchical clustering

#import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

X = cluster1

points = X[:,[0,1]]

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
# create clusters
hc = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(points)

plt.scatter(points[y_hc ==0,0], points[y_hc == 0,1], s=100, c='red')
plt.scatter(points[y_hc==1,0], points[y_hc == 1,1], s=100, c='black')
plt.scatter(points[y_hc ==2,0], points[y_hc == 2,1], s=100, c='blue')
#plt.scatter(points[y_hc ==3,0], points[y_hc == 3,1], s=100, c='cyan')

##Evaluation methods for different clustering techniques
###!!VARIABLE y_km changes for each of the existing techniques, y_km is example variable
###for KMeans. Respective variable for
###DBSCAN is labels
###Hierarchical clustering is y_hc
from sklearn.metrics import silhouette_score
f'Silhouette Score(n=2): {silhouette_score(X, y_km)}'

##X[:,2] is base data column with true class labels
from sklearn.metrics.cluster import normalized_mutual_info_score as mi
mi(X[:,2],y_km)
