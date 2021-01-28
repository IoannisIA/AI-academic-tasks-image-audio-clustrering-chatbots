from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

######################################################################################################

train = pd.read_csv("train.csv")

# Separating out the features
x = train.iloc[:,1:].values

# Separating out the label
y = train.loc[:,['label']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=300)
principalComponents = pca.fit_transform(x)
principalDf300 = pd.DataFrame(data=principalComponents)

DIGITS_PCA_300 = pd.concat([principalDf300, train[['label']]], axis = 1)

######################################################################################################################

## Separating the X and Y variable

y = DIGITS_PCA_300['label']

## Dropping the variable 'label' from X variable
X = DIGITS_PCA_300.drop(columns='label')

# train test split and keep 25% for testing

(trainData, testData, trainLabels, testLabels) = train_test_split(X, y, test_size=0.25, random_state=42)

# now, let's take 10% of the training data and use that for validation

(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)


######################################################################################################


# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k

kVals = range(1, 30, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier

for k in range(1, 30, 2):
    # train the k-Nearest Neighbor classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)
    # evaluate the model and update the accuracies list
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# find the value of k that has the largest accuracy

i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                       accuracies[i] * 100))

# re-train our classifier using the best k value and predict the labels of the
# test data

model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)
predictions = model.predict(testData)
# print(predictions[1])

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits

print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

print("Confusion matrix")
print(confusion_matrix(testLabels, predictions))
