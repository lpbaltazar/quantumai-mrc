import matplotlib as mpl
mpl.use('Tkagg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def visualize(df, labels, filecode):
	include_cols = []
	columns = input('Enter numerical columns:')
	include_cols.append(columns)
	df = df[include_cols]

	labels = labels.astype('str')
	df['labels'] = labels

	plot = sns.pairplot(data=df, vars=include_cols, hue = 'labels')
	plot.savefig('plot-' + filecode + '.jpg')
