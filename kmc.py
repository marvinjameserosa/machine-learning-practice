import numpy as np
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

# Data processing
rawData = load_digits()
data = scale(rawData.data)
y = rawData.target

# Amount of centroids 
## If you want to set it manually
k = 10
## If you want a more dynamic
### k = len(np.unique(y))

sample, features = data.shape

# Classifier Function
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

# Scoring
clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)