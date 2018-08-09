import matplotlib as mpl
mpl.use('Tkagg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


file = 'final_preprocessed.csv'
dcols = {'Province':np.str}
cols = ['CustomerID', 'AccountType', 'CustomerType', 'NoOfProducts', 
		'NumOfServicesAvailed', 'GC', 'CD', 'R', 'CS',
		'NumOfServicesAvailedOutofWarranty', 'AverageBillAmountOutOfWarranty',
		'AverageBillAmount', 'Class']
X = pd.read_csv(file, encoding="ISO-8859-1", dtype = dcols, usecols= cols)
X = X.set_index('CustomerID')

X = X.values

def plotClusters(centers, x1, x2, cluster):
	plt.figure()
	plt.scatter(x1,x2, c='black', s= 7)
	plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

	for i, c in enumerate(centers):
	    plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
	                s=50, edgecolor='k')

	outfile = 'results/result-' + str(cluster) +'.jpg'
	plt.savefig(outfile)

range_n_clusters = [2, 3, 4, 5, 6]

sse = {}
for k in range_n_clusters:
	print('k:', k)
	kmeans = KMeans(k).fit(X)
	labels = kmeans.predict(X)
	centers = kmeans.cluster_centers_

	#plotClusters(centers, x1, x2, k)

	sse[k] = kmeans.inertia_
	print(sse[k])
	plt.figure()
	plt.plot(list(sse.keys()), list(sse.values()))
	plt.xlabel('Cluster')
	plt.ylabel('Sum of squared Errors of prediction')
	outfile= 'results/elbowmethod-result.jpg'
	plt.savefig(outfile)
