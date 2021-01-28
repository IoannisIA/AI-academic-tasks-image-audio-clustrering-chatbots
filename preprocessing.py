# https://www.kaggle.com/saurabh105/pca-visualization-on-mnist-data-from-scratch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh

data = pd.read_csv("train.csv")


label = data.label.astype(np.int)
data.drop("label", axis=1, inplace=True)

standardized_data = StandardScaler().fit_transform(data)

sample_data = standardized_data
covariance_matrix = np.matmul(sample_data.T, sample_data)


values, vectors = eigh(covariance_matrix)
print("Last 10 eigen values:")
print(values[:][-10:])
print("\nCorresponding vectors:")
print(vectors[-10:])

values = values[-2:]
vectors = vectors[:,-2:]
vectors = vectors.T
print("Shape of eigen value: ", values.shape)
print("Shape of eigen vectors: ", vectors.shape)


reduced_data = np.matmul(vectors, sample_data.T)
print("Reduced data shape: ", reduced_data.shape)

reduced_data = np.vstack((reduced_data, label))
reduced_data = reduced_data.T

reduced_df = pd.DataFrame(reduced_data, columns=['X', 'Y', 'label'])
reduced_df.label = reduced_df.label.astype(np.int)
print(reduced_df.head())


g = sns.FacetGrid(reduced_df, hue='label', height=12).map(plt.scatter, 'X', 'Y').add_legend()

plt.show()
