import csv
import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import timeit
import sys

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#import seaborn as sns
import matplotlib.pyplot as plt

# scoring
from sklearn.metrics import davies_bouldin_score, silhouette_score

if len(sys.argv) != 3:
    raise ValueError('Please provide a data file and the bumber of clusters')

root_dir=sys.argv[1]#'RawData/AllBooks_baseline_DTM_Labelled.csv'
numKs = int(sys.argv[2])

fullset = pd.read_csv(root_dir)

#split test train
train, test= train_test_split(fullset, test_size=0.30, random_state=42)

#build kmeans model
kmModel = KMeans(n_clusters=numKs)
kmedoidsModels = KMedoids(n_clusters=numKs) 
hierarchyModels = AgglomerativeClustering(n_clusters=numKs)
#trainedKM = [kmModel[i].fit(train.iloc[: , 1:]) ]

print("run time \n")

trained,trainedMed,predsMed,trainedHie,predsHie  = [],[],[],[],[]
preds = []

start = timeit.default_timer()
trained.append(kmModel.fit(train.iloc[: , 1:]))
preds.append(trained[0].predict(test.iloc[: , 1:]))
stop = timeit.default_timer()
#print(str(stop-start))

start = timeit.default_timer()
trainedMed.append(kmedoidsModels.fit(train.iloc[: , 1:]))
predsMed.append(trainedMed[0].predict(test.iloc[: , 1:]))
stop = timeit.default_timer()
#print(str(stop-start))

start = timeit.default_timer()
trainedHie.append(hierarchyModels.fit(train.iloc[: , 1:]))
predsHie.append(trainedHie[0].fit_predict(test.iloc[: , 1:]))
stop = timeit.default_timer()
print(str(stop-start))
    
corrrectLabels = []

for i in range(len(test.iloc[: , 0])):
    tmp = test.iloc[i , 0]
    tmp=tmp[0:int(tmp.find('_'))]
    corrrectLabels.append(tmp)
print("Number of unique books = " + str(len(set(corrrectLabels))))

shs, db = {}, {}

print("solhouette score : \n")
shs=silhouette_score(test.iloc[:,1:], predsMed[0])
print(shs)
print("davies_bouldin score : \n")
db=davies_bouldin_score(test.iloc[:,1:], predsMed[0])
print(db)