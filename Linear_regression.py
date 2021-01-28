from sklearn.linear_model import SGDRegressor
import sys
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

X = df[['temp']]

y = df['cnt']

old_stdout = sys.stdout

sys.stdout = mystdout = StringIO()

n = 1 ;# number of instances to keep
fX = X[:n].values
fy = y[:n].values

learning_rate = 2
sgd = SGDRegressor(loss="squared_loss", learning_rate='constant', eta0=learning_rate, penalty=None, max_iter=1,
                   average=False, random_state=2018, verbose=1)

p_sum = [] ;# this holds the sum of y-y_hat, for all instances (ok, we habe only one instance)
epochs = 15 ;# number of epochs

for epoch in range(epochs):
    model = sgd.partial_fit(fX, fy)
    y = model.predict(fX)
    p_sum.append(np.sum(fy-y))



sys.stdout = old_stdout
loss_history = mystdout.getvalue()
print(loss_history)
loss_list = []
for line in loss_history.split('\n'):
    if(len(line.split("loss: ")) == 1):
        continue
    loss_list.append(float(line.split("loss: ")[-1]))


print(model.coef_)


plt.figure()
plt.plot(np.arange(len(loss_list)), loss_list)
plt.scatter(np.arange(len(loss_list)), loss_list)
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


