#!/usr/bin/env python

"""Train classifiers to predict MNIST data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from math import pi
import sklearn.preprocessing


def get_data(dataset='wbcd'):

    if dataset == 'wbcd':
        df = pd.read_csv('data.csv')
        df = df.dropna(axis='columns', how='all')
        #y = df['diagnosis']
        #le = sklearn.preprocessing.LabelEncoder()
        #le.fit(y)
        #y = le.transform(y)

        B = df.iloc[19,:]
        M = df.iloc[18,:]

        # split dataframe into two based on diagnosis
        dfM=df[df['diagnosis'] =='M']
        dfB=df[df['diagnosis'] =='B']
        df = df.drop(labels='diagnosis',axis=1)
        df_mean = df.loc[:,df.columns.str.endswith('mean')]
        #features = [s[:-5] for s in list(df_mean.columns)] 
        #groups = ['mean', 'se', 'worst']
        dfB = dfB.drop(labels='diagnosis', axis=1)

        dfBm = dfB.loc[:,dfB.columns.str.endswith('mean')]
        dfBm_norm = (dfBm-dfBm.min())/(dfBm.max()-dfBm.min())
        #benigndata_mean = dfBm_norm.mean().values
        Bmean = B.iloc[2:12]
        Bmean_norm = (Bmean-dfBm.min())/(dfBm.max()-dfBm.min())
        #patientdata_mean = Bmean_norm.values

        dfBs = dfB.loc[:,dfB.columns.str.endswith('se')]
        dfBs_norm = (dfBs-dfBs.min())/(dfBs.max()-dfBs.min())
        #benigndata_se = dfBs_norm.mean().values
        Bse = B.iloc[12:22]
        Bse_norm = (Bse-dfBs.min())/(dfBs.max()-dfBs.min())
        #patientdata_se = Bse_norm.values

        dfBw = dfB.loc[:,dfB.columns.str.endswith('worst')]
        dfBw_norm = (dfBw-dfBw.min())/(dfBw.max()-dfBw.min())
        #benigndata_worst = dfBw_norm.mean().values
        Bworst = B.iloc[22:32]
        Bworst_norm = (Bworst-dfBw.min())/(dfBw.max()-dfBw.min())
        #patientdata_worst = Bworst_norm.values

        data = {'features': [s[:-5] for s in list(df_mean.columns)],
                'groups': ['mean', 'se', 'worst'],
                'benigndata_mean': dfBm_norm.mean().values,
                'patientdata_mean': Bmean_norm.values,
                'benigndata_se': dfBs_norm.mean().values,
                'patientdata_se': Bse_norm.values,
                'benigndata_worst': dfBw_norm.mean().values,
                'patientdata_worst': Bworst_norm.values
        }
    else:
        raise NotImplemented()
    return data


def createRadar(stats,patientdata):
    # number of variables
    N = len(features)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    stats = np.concatenate((stats,[stats[0]]))
    patientdata = np.concatenate((patientdata,[patientdata[0]]))
    angles=np.concatenate((angles,[angles[0]]))
 
    # Initialise the spider plot
    fig, axes = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    axes = axes.ravel()
    for idx,ax in enumerate(axes):
        ax.plot(angles, stats, '-', linewidth=2)
        ax.fill(angles, stats, alpha=0.25)
        ax.plot(angles, patientdata, 'o-', linewidth=2)
        ax.set_thetagrids(angles * 180/np.pi, features)
        ax.set_yticklabels([])
        ax.set_title(groups[0], fontsize=16)
        ax.grid(True)


if __name__ == '__main__':
    print(get_data('wbcd'))
