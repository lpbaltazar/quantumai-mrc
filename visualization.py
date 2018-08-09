import matplotlib as mpl
mpl.use('Tkagg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

numeric_cols = ['NoOfProducts', 'AverageBillAmount', 'NumOfServicesAvailed',
		'AverageBillAmountOutOfWarranty', 'NumOfServicesAvailedOutofWarranty',
		'NumOfComplaints']

def visualize(df, labels, filecode):
	present_cols = df.columns.values

	include_cols = []
	for col in numeric_cols:
		if col in present_cols:
			include_cols.append(col)

	df = df[include_cols]
	df = df.copy()

	labels = labels.astype('str')
	df.loc[:, 'Cluster'] = labels

	plot = sns.pairplot(data=df, vars=include_cols, hue = 'Cluster', diag_kind="hist")
	plot.savefig('results/visualization/visualizeplot-' + filecode + '.jpg')
	print('Saved plot!')