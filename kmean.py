import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

from sklearn.cluster import KMeans
l = 40 #lambda
aic_results = [0] * (26-15)
for k in range(15,26):
    print("\n\nk=" + str(k))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.show()
    
    test = [[] for x in range(k)]
    centers = [0] * k
    
    #add all points to its cluster
    for i in range(len(y_true)):
        h = y_kmeans[i]
        test[h].append(X[i])
        
    #calc cluster center
    for x in range(k):
        s = 0
        for y in range(len(test[x])):
            s = s + test[x][y]
        s = s/len(test[x])
        centers[x] = s
        
    #calc RSS for each cluster
    rss = [0] * k
    for x in range(k):
        rss[x] = 0
        for y in range(len(test[x])):
            d = abs(centers[x] - test[x][y])
            c = d[0] * d[0] + d[1] * d[1]
            rss[x] = rss[x] + c
            
    #calc RSS median
    rss_m = 0
    for x in range(k):
        rss_m = rss_m + rss[x]
    rss_m = rss_m/k
    print("rss_median: " + str(rss_m))
    
    #calc RSS_k + l*k
    aic = rss_m + l*k
    aic_results[k-15] = aic
    print("aic: " + str(aic))
    
kmin = aic_results.index(min(aic_results))
kmin += 15
print("\nThe minimal value is calculated for k = " + str(kmin))

