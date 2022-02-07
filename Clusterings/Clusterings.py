import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('musteriler.csv')


x = veriler.iloc[:,3:].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values.ravel() #bağımlı değişken

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init = 'k-means++')
kmeans.fit(x)

sonuclar = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++',random_state=123)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar) # dirsek noktasi optimum nokta
plt.show() # 4 bulundu

kmeans = KMeans(n_clusters=4, init = 'k-means++',random_state=123)
y_tahmin = kmeans.fit_predict(x)

plt.scatter(x[y_tahmin==0,0],x[y_tahmin==0,1],s=100,c="red")
plt.scatter(x[y_tahmin==1,0],x[y_tahmin==1,1],s=100,c="blue")
plt.scatter(x[y_tahmin==2,0],x[y_tahmin==2,1],s=100,c="green")
plt.scatter(x[y_tahmin==3,0],x[y_tahmin==3,1],s=100,c="cyan")
plt.show()