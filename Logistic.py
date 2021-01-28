from sklearn.linear_model import SGDClassifier
import sys
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

# X = df[['season','yr','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum']]
X = df[['temp']]
y = df['cnt']

labels = ["0", "1"]
y_categorical = pd.cut(y, 2, labels=labels)
print(y_categorical.head())

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

n = 1 ;# number of instances to keep
fX = X[:n].values
fy = y_categorical[:n].values

learning_rate = 2


sgd  = SGDClassifier(loss="squared_loss", learning_rate='constant', eta0=learning_rate, penalty=None, max_iter=1,
                     average=False, random_state=2018, verbose=1)
sgd2 = SGDClassifier(loss="log", learning_rate='constant', eta0=learning_rate, penalty=None, max_iter=1,
                     average=False, random_state=2018, verbose=1)



p_sum  = [] ;# this holds the sum of y-y_hat, for all instances (ok, we have only one instance)
p2_sum = []
epochs = 15 ;# number of epochs

for epoch in range(epochs):
    model = sgd.partial_fit(fX, fy, classes=labels)
    model2 = sgd2.partial_fit(fX, fy, classes=labels)
    y = model.predict(fX)
    p = model.decision_function(fX)
    c = (1 - np.mean(y == fy))
    p_sum.append(p)
    y2 = model2.predict(fX)
    p2 = model2.decision_function(fX)
    c2 = (1 - np.mean(y2 == fy))
    p2_sum.append(p2)



sys.stdout = old_stdout
loss_history = mystdout.getvalue()
#print(loss_history)
loss_list  = []
loss2_list = []
use_first = True
for line in loss_history.split('\n'):
    if(len(line.split("loss: ")) == 1):
        continue
    if use_first:
        loss_list.append(float(line.split("loss: ")[-1]))
    else:
        loss2_list.append(float(line.split("loss: ")[-1]))
    use_first = not use_first


plt.figure()
plt.plot(np.arange(len(loss_list)), loss_list)
plt.scatter(np.arange(len(loss_list)), loss_list)
plt.xlabel("Time in epochs")
plt.ylabel("Loss")
plt.show()
plt.figure()
plt.plot(np.arange(len(loss2_list)), loss2_list)
plt.scatter(np.arange(len(loss2_list)), loss2_list)
plt.xlabel("Time in epochs")
plt.ylabel("Loss")
plt.show()


plt.figure()
plt.plot(p_sum, loss_list)
plt.scatter(p_sum, loss_list)
plt.scatter(p_sum[:1], loss_list[:1], color='red')
plt.xlabel("$y-\hat{y}$")
plt.ylabel("Loss")
plt.show()
plt.figure()
plt.plot(p2_sum, loss2_list)
plt.scatter(p2_sum, loss2_list)
plt.scatter(p2_sum[:1], loss2_list[:1], color='red')
plt.xlabel("$y-\hat{y}$")
plt.ylabel("Loss")
plt.show()


