import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

center_list = []
kmiterlist = []
kmeans_list = [20, 30, 40, 50, 60]

purity_list = []
for i in range(0, len(kmeans_list)):

    km = KMeans(n_clusters=kmeans_list[i], init="random", n_init=1).fit(principalDf300)

    print("kmeans_number_of_iterations :", km.n_iter_)
    kmiterlist.append(km.n_iter_)

    y_kmeans = km.predict(principalDf300)
    df_temp = pd.DataFrame(y, y_kmeans, columns=["True"]).reset_index()

    majority_count_value = 0

    for i in df_temp["index"].unique():
        df_inner = df_temp[df_temp["index"] == i]
        count_true_val = df_inner["True"].value_counts()
        majority_count_value += count_true_val.iloc[0]

    majority_purity_label = majority_count_value / 42000
    purity_list.append(majority_purity_label)


"""
Purity Explained
Intuitive Definition of Purity: You can define Purity of a cluster as follows:
If a cluster contains all points in the same class, its purity is 1 (highest).
If each class is equally present in a cluster, its purity is 0.
"""

sns.scatterplot(kmeans_list, purity_list)
plt.title("K-values vs Purity")
plt.show()
