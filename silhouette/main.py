import pandas as pd
import numpy as np
import os

from sklearn.cluster import KMeans, AgglomerativeClustering

from kmeans import analysis

file = os.path.abspath('../preprocessed_final.csv')
print(file)
dcols = {'Province':np.str}
#remove nrows if plan to run using whole dataset
X = pd.read_csv(file, encoding="ISO-8859-1", dtype = dcols)
new_col = pd.factorize(X['Province'])
X.drop(['Province'], inplace = True, axis = 1)
province = pd.Series(data = new_col[0], name="Province")
X = pd.concat([X, province], axis = 1)

X = X.set_index('CustomerID')

algo = 'kmeans'
analysis(X.values, algo)