import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram,linkage
df = pd.read_csv('/kaggle/input/milletprotein/nDataMillet.csv')
X = df[['proteins']]
Z = linkage(X, method = 'ward')
plt.figure(figsize = (10,4))
plt.title('DENDROGRAM')
plt.xlabel('PROTEIN')
plt.ylabel('DISTANCE')
dendrogram(Z)
plt.show()
n_clusters = 3
hc = AgglomerativeClustering(n_clusters = n_clusters , affinity = 'euclidean' , linkage = 'ward')
y_hc = hc.fit_predict(X)
df['Cluster'] = y_hc
df_sorted = df.sort_values(by=['Cluster' , 'proteins'] , ascending = [True , False])
print(df_sorted)
silhouette_score = silhouette_score(X,y_hc)
print("Extracted Protein:" , X)
print("Silhouette Score:" , silhouette_score)
