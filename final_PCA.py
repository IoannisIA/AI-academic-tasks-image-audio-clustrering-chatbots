import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

"""
The next step is to rescale our data to a range between 0 and 1, 
with MinMaxScaler from sklearn. Data sets can contains random 
attributes with different units of measurement, one attribute can be price, 
another can be percentage, another can be a binary value. 
To perform a reasonable covariance analysis we need to normalize this data, 
putting all attributes in the same unit of measurement (the range between 0 and 1), 
it will improve the maximization of the variance for each component that our 
PCA needs to perform its matrix operations in the best way.
"""
data = np.genfromtxt('train.csv', delimiter=',', dtype='float64')
scaler = MinMaxScaler(feature_range=[0, 1])
data_rescaled = scaler.fit_transform(data[1:, 1:784])

"""
This plot tells us that selecting 300 components we can preserve something 
around 98.8% or 99% of the total variance of the data. 
It makes sense, we’ll not use 100% of our variance, because it denotes all components, 
and we want only the principal ones.

"""
#Fitting the PCA algorithm with our Data
pca = PCA().fit(data_rescaled)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Features')
plt.ylabel('Variance (%)') #for each component
plt.title('MNIST Dataset Variance')
plt.show()

"""
The dataset variable will store our new data set, 
now with all the samples but 300 features.

"""
pca = PCA(n_components=300)
dataset = pca.fit_transform(data_rescaled)

"""
You can use the dataset as your X, and the data[:, 0:1] as your Y, 
to split in train and test data, and to fit in your models, 
reducing the overfitting and the training time.
"""

