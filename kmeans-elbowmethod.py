import matplotlib as mpl
mpl.use('Tkagg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from visualization import visualize

file = 'preprocessed_final.csv'
dcols = {'Province':np.str}
#change the cols according to the features assigned to you
cols = ['CustomerID', 'AccountType', 'CustomerType', 'NoOfProducts', 
		'NumOfServicesAvailed', 'GC', 'CD', 'R', 'CS',
		'FirstTime', 'Province',
		'NumOfComplaints', 'Class']
df = pd.read_csv(file, encoding="ISO-8859-1", dtype = dcols, usecols= cols)
new_col = pd.factorize(df['Province'])
df.drop('Province', inplace = True, axis = 1)
province = pd.Series(data = new_col[0], name="Province")
df = pd.concat([df, province], axis = 1)
df = df.set_index('CustomerID')

range_n_clusters = [2, 3, 4, 5, 6]

# change attributes according to the features assigned to you
attributes = ['AccountType', 'CustomerType', 'NoOfProducts', 
		'NumOfServicesAvailed', ['GC', 'CD', 'R', 'CS'],
		'FirstTime', 'Province',
		'NumOfComplaints', 'Class']

code_dictionary = list(zip(range(0,10), attributes))
with open("lei_code_dictionary.txt", "w") as f:
    for code in code_dictionary:
        f.write(str(code) +"\n")

combination = list(set(itertools.combinations(range(0, 9), 5)))
for i in range(1):
	cols = []
	comb = combination[i]
	for j in comb:
		if type(attributes[j]) == list:
			cols.extend(attributes[j])
		else:
			cols.append(attributes[j])
	a = map(str, comb)    
	file_code = ''.join(a)     
	X = df.loc[:, df.columns.isin(cols)]
	print('File Code: ',file_code)
	sse = {}
	for k in range_n_clusters:
		print('k:', k)
		kmeans = KMeans(k, random_state = 42).fit(X.values)
		labels = kmeans.predict(X.values)
		centers = kmeans.cluster_centers_

		sse[k] = kmeans.inertia_
		print(sse[k])
		plt.figure()
		plt.plot(list(sse.keys()), list(sse.values()))
		plt.xlabel('Cluster')
		plt.ylabel('Sum of squared Errors of prediction')
		outfile= 'results/elbow-plot/kmeans-elbowmethod-result'+'-'+file_code+'.jpg'
		plt.savefig(outfile)

	optimal = int(input('Enter optimal number of clusters: '))
	kmeans = KMeans(optimal, random_state = 42)
	labels = kmeans.fit_predict(X.values)
	visualize(df, labels, file_code)
	cluster_labels = pd.DataFrame(labels, index=X.index, columns = ['Cluster_Labels'])
	cluster_labels.to_csv('results/labels/labels'+'-'+file_code+'.csv',sep=',', encoding='utf-8', index='True')
