import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import call
from sklearn.model_selection import KFold
from sklearn import metrics

data = pd.read_csv("data.csv")

features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

bins = [0, 2000, 3000, 4000, 8000, 9000]

labels = ['ab', 'ac', 'ad', 'ae', 'af']

df = pd.DataFrame(data=data)

df['cnt'] = pd.cut(df.cnt, bins=bins, labels=labels)

X = df[features].values

y = df.cnt.values

kf = KFold(n_splits=20, shuffle=False, random_state=None)
avg_acc_train = 0.0
avg_acc = 0.0
best_acc = 0.00
best_clf = None
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    acc_train = metrics.accuracy_score(y_train, y_pred)
    y_pred = clf.predict(X_test)
    acc_test = metrics.accuracy_score(y_test, y_pred)
    avg_acc_train += acc_train / 20
    avg_acc += acc_test / 20

    if acc_test > best_acc:
        best_clf = clf

print(acc_test)

export_graphviz(best_clf, 'tree.dot', filled=True, rounded=True, special_characters=True, feature_names=features, class_names=labels)

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

