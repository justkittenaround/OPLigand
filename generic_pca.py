
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
from PIL import Image
import time
import copy
import h5py
import pandas as pd
from scipy import stats

import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



R_DATA_PATH = 'receptor.npy'
L_DATA_PATH = 'ligands.npy'
##############################


def get_data():
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    rdata = np.load(R_DATA_PATH)
    ldata = np.load(L_DATA_PATH)
    print('# of receptors:', len(receptor_data))
    return rdata, ldata


d = np.append(rdata, ldata, axis=0)

df = pd.DataFrame(np.asarray(d))
df = df.fillna(0)


color = np.append(np.ones(len(rdata)), np.zeros(len(ldata)), axis=0)
df.insert(loc=0, column='color', value=color)


##2d Plot
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[df.keys()[0:]].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
plt.close()
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue='color',
    palette=sns.color_palette("bright", 2),
    data=df,
    legend="full",
    alpha=0.3
)
plt.title('2d PCA Hidden States per Class')
plt.show()


##3d Plot
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[df.keys()[1:]].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
plt.close()
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[:,"pca-one"],
    ys=df.loc[:, "pca-two"],
    zs=df.loc[:, "pca-three"],
    c=df['color'],
    cmap='Spectral'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.title('3d PCA Hidden States per Class')
plt.show()








########################################################33
