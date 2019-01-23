#%%
#Loading data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
dfall = pd.read_csv("Medicare_Provider_Util_Payment_PUF_CY2016.txt",sep = "\t", skiprows = 2, header = None)

#%%
#dfall[24].corr(dfall[25])
#dfall[23].corr(dfall[25])
#dfall[22].corr(dfall[25])
#plt.hist(dfall[21])
#plt.hist(dfall[25])
#Choosing four columns to analyze
df = pd.read_csv("Medicare_Provider_Util_Payment_PUF_CY2016.txt",sep = "\t", usecols = (20,21,23,25), skiprows = 2, header = None)

#%%
#Prepocessing
df = df[df[20] < 1000]
df = df[df[21] < 1000]
df = df[df[25] < 2000]
df = df[df[23] < 4000]
    
dfnew = df.values
    
#Standardize
scaler = StandardScaler()
df_sc = scaler.fit_transform(dfnew)
    
df_sctenk = df_sc[0:10000,:]

#Plot on 2D
df_embedded = TSNE(n_components = 2).fit_transform(df_sctenk)

plt.figure()
plt.scatter(df_embedded[:,0],df_embedded[:,1], marker = ".")

#%%
#Choosing the best number of clusters
score = np.zeros(9)
for i in range(2,11):
    mkmeans = MiniBatchKMeans(n_clusters = i, random_state = 0).fit(df_sc)
    score[i-2] = mkmeans.score(df_sc)

plt.figure()
plt.plot(range(2,11),-score[:])
plt.xlabel('# of clusters')
plt.ylabel('objective')

#%%
#We choose to use 5 clusters
kmeans = KMeans(n_clusters = 5, random_state = 0).fit(df_sc)

#Plot the 5 clusters for the first 10,000 rows
plt.figure()
for i in range(10000):
    if kmeans.labels_[i] == 0:
        plt.scatter(df_embedded[i,0], df_embedded[i,1], marker = '.', c = 'g')
    elif kmeans.labels_[i] == 1:
        plt.scatter(df_embedded[i,0], df_embedded[i,1], marker = '.', c = 'r')
    elif kmeans.labels_[i] == 2:
        plt.scatter(df_embedded[i,0], df_embedded[i,1], marker = '.', c = 'b')
    elif kmeans.labels_[i] == 3:
        plt.scatter(df_embedded[i,0], df_embedded[i,1], marker = '.', c = 'y')
    else:
        plt.scatter(df_embedded[i,0], df_embedded[i,1], marker = '.', c = 'c')

#Centroid coordinates
print(scaler.inverse_transform(kmeans.cluster_centers_))

#Silhouette score for clustering
print(silhouette_score(df_sc[0:10000,:], kmeans.labels_[0:10000]))       

#%%
#Distance of each data point to its centroid
dis = np.zeros(len(df_sc))
for i in range(len(df_sc)):
    dis[i] = np.linalg.norm(df_sc[i] - kmeans.cluster_centers_[kmeans.labels_[i]])
#np.mean(dis)
#np.std(dis)
#%%
#Function for incoming data entries on payment amount
def testing_function(Xmatrix, scaled = True, scaling = scaler, centroids = kmeans.cluster_centers_):
# Return: None if the point is outlier
#         True/False if the payment is overpriced
    clusterprice = np.array([128.25, 271.62, 104.51, 1016.67, 155.89])
    group = np.zeros(len(Xmatrix))
    dist = np.zeros(5)
    for i in range(len(Xmatrix)):
        dist[0] = np.linalg.norm(Xmatrix[i] - kmeans.cluster_centers_[0])
        dist[1] = np.linalg.norm(Xmatrix[i] - kmeans.cluster_centers_[1])
        dist[2] = np.linalg.norm(Xmatrix[i] - kmeans.cluster_centers_[2])
        dist[3] = np.linalg.norm(Xmatrix[i] - kmeans.cluster_centers_[3])
        dist[4] = np.linalg.norm(Xmatrix[i] - kmeans.cluster_centers_[4])
        if min(dist) > 2.2:
            group[i] = None
        else:
            index = np.argmin(dist)
            if scaled:
                realprice = scaling.inverse_transform(Xmatrix[i,:])[3]
            else:
                realprice = Xmatrix[i,3]
            group[i] = (realprice > clusterprice[index])
    return group

print(testing_function(np.random.uniform(size = (1000,4))*4-2))
