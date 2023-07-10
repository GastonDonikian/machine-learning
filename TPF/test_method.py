import numpy as np #linear algebra 
import pandas as pd #creating and manipulating dataframes
import matplotlib.pyplot as plt #visuals
import seaborn as sns #visuals

from sklearn.cluster import KMeans #K-Means
from sklearn.cluster import DBSCAN #DBSCAN

from sklearn.preprocessing import StandardScaler #scaler

#read the data
blobs = pd.read_csv('./resources/cluster_blobs.csv')
moons = pd.read_csv('./resources/cluster_moons.csv')
circles = pd.read_csv('./resources/cluster_circles.csv')


#rename moons columns 
moons.rename({"X_1": 'X1', 'X_2': 'X2'}, axis = 1, inplace = True)

#define the function 
def display_categories(model,data):
    labels = model.fit_predict(data)
    plt.figure(figsize = (8,4), dpi = 100)
    sns.scatterplot(data=data,x='X1',y='X2',hue=labels,palette='Set1')
    plt.show()

def display_datasets():
    # Blobs
    plt.figure(figsize = (8,4), dpi = 100)
    sns.scatterplot(data=blobs,x='X1',y='X2')
    plt.show()
    # Moons
    plt.figure(figsize = (8,4), dpi = 100)
    sns.scatterplot(data=moons,x='X1',y='X2')
    plt.show()
    # Circles
    plt.figure(figsize = (8,4), dpi = 100)
    sns.scatterplot(data=circles,x='X1',y='X2')
    plt.show()


def run_k_means():
    model = KMeans(n_clusters = 2)
    display_categories(model,moons)

    model = KMeans(n_clusters = 3)
    display_categories(model,blobs)

    model = KMeans(n_clusters = 2)
    display_categories(model,circles)

def run_dbscan():
    model = DBSCAN(eps=0.65)
    display_categories(model,blobs)

    model = DBSCAN(eps=0.10)
    display_categories(model,moons)

    model = DBSCAN(eps=0.10)
    display_categories(model,circles)


if __name__ == "__main__":
    #display_datasets()
    run_k_means()
    #run_dbscan()
