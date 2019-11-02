#!/usr/bin/env python

"""Train classifiers to predict MNIST data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
from math import pi
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_data(id_loc, dataset='wbcd'):

    if dataset == 'wbcd':
        df = pd.read_csv('data.csv')
        df = df.dropna(axis='columns', how='all')
        y = df['diagnosis']

        B = df.iloc[id_loc,:]

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
                'id': df.iloc[19,:][0].astype(int),
                'gpnames':['mean', 'se', 'worst'],
                'groups': [{'benigndata': dfBm_norm.mean().values,
                                'patientdata': Bmean_norm.values},
                            {'benigndata': dfBs_norm.mean().values,
                                'patientdata': Bse_norm.values},
                            {'benigndata': dfBw_norm.mean().values,
                                'patientdata': Bworst_norm.values}]
                }
        df = df.drop(labels='id', axis=1)
        scaled_data = StandardScaler().fit(df).transform(df)
        pca = PCA(n_components = 2).fit(scaled_data)
        x_pca = pca.transform(scaled_data)

    else:
        raise NotImplemented()
    return data, df, y, pca, x_pca


def createRadar(data, title):
    # number of variables
    N = len(data['features'])
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles=np.concatenate((angles,[angles[0]]))
 
    # Initialise the spider plot
    fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(6, 6), 
                                subplot_kw=dict(polar=True))
    axes = axes.ravel()
    for idx,ax in enumerate(axes):
        ax.figure
        stats = data['groups'][idx]['benigndata']
        stats = np.concatenate((stats,[stats[0]]))
        ax.plot(angles, stats, '-', linewidth=2)
        ax.fill(angles, stats, alpha=0.25)
        pdat = data['groups'][idx]['patientdata']
        pdat = np.concatenate((pdat,[pdat[0]]))
        ax.plot(angles, pdat, 'o-', linewidth=2)
        ax.set_thetagrids(angles * 180/np.pi, data['features'])
        ax.set_yticklabels([])
        ax.set_title(data['gpnames'][idx], fontsize=16)
        ax.grid(True)
    plt.suptitle(title, y=1.02, horizontalalignment='center',
                    verticalalignment='top', fontsize=16)
    plt.tight_layout()
    plt.show()


def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)


def createBiplot(df, y, pca, x_pca):
    components = pd.DataFrame(pca.components_.T, index=df.columns, columns=['PCA1','PCA2'])
    ax = plt.subplot(111, aspect='equal')
    for di in np.unique(y):
        xpca = x_pca[y==di]
        # confidence ellipse
        comp1_mean = np.mean(xpca[:,0])
        comp2_mean = np.mean(xpca[:,1])
        cov = np.cov(xpca[:,0], xpca[:,1])
        # main scatterplot
        plt.scatter(xpca[:,0], xpca[:,1], c=di, cmap='plasma', alpha=0.4, edgecolors='black', s=40)
        e = get_cov_ellipse(cov, (comp1_mean, comp2_mean), 3, fc=di, alpha=0.4)
        ax.add_artist(e)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.ylim(15,-15)
    plt.xlim(20,-20)
    # individual feature values
    ax2 = plt.twinx().twiny()
    ax2.set_ylim(-0.5,0.5)
    ax2.set_xlim(-0.5,0.5)
    # reference lines
    ax2.hlines(0,-0.5,0.5, linestyles='dotted', colors='grey')
    ax2.vlines(0,-0.5,0.5, linestyles='dotted', colors='grey')
    # offset for labels
    offset = 1.07
    # arrow & text
    for a, i in enumerate(components.index):
        ax2.arrow(0, 0, components['PCA1'][a], components['PCA2'][a], \
                    alpha=0.5, facecolor='white', head_width=.01)
        ax2.annotate(i, (components['PCA1'][a]*offset, components['PCA2'][a]*offset),
                            color='orange')
    plt.show()

if __name__ == '__main__':
    d, df, y, pca, x_pca = get_data(18)
    #title = 'Breast Cancer Diagnosis Radar\nPatient ID: {}'.format(d['id'])
    #createRadar(d, title)
    createBiplot(df, y, pca, x_pca)


