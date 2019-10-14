'''
https://www.kaggle.com/mirichoi0218/classification-breast-cancer-or-not-with-15-ml
 - a rendering of miri choi's blog on
[Classification] Breast Cancer or Not (with 15 ML) in python
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

data = pd.read_csv("data.csv")
print('Shape of data: ', data.shape)
print('-'*80); print(data.head())
print('-'*80); print(data.isna().sum())
filtered_data = data.dropna(axis='columns', how='all')
print('-'*80); print('Shape of filtered data: ', filtered_data.shape)
print('-'*80); print(filtered_data.describe())
filtered_data['diagnosis'] = pd.Categorical(filtered_data['diagnosis']).copy()
filtered_data['diagnosis'] = filtered_data.diagnosis.cat.codes
print('-'*80); print(filtered_data.dtypes)
'''
The field diagnosis has either B (beningn) or M (malignant) value.
Let’s check how many patients are in each category.
'''
print('-'*80); print('Categories:\n', filtered_data['diagnosis'].value_counts())
'''
https://github.com/InsightDataLabs/ipython-notebooks/blob/master/seaborn.ipynb
Copied mostly from this link:
https://stackoverflow.com/questions/56768118/titles-for-histograms-on-diagonal-when-using-seaborn-pairgrid-in-python-for-gene
https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another
'''
def corrplot(df):
    def scatterfunc(x, y, **kws):
        plt.scatter(x, y, linewidths=1, facecolor="k", s=10, alpha = 0.5)
        spline = np.polyfit(x, y, 5)
        model = np.poly1d(spline)
        x = np.sort(x)
        plt.plot(x,model(x),'r-')
        
    def histfunc(x, **kws):
        plt.hist(x,bins=30,color = "black", ec="white")

    def corrfunc(x, y, **kws):
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),
                    xy=(.3, .5), xycoords=ax.transAxes)
    
    def make_diag_titles(g,titles):
        for ax in g.axes.flatten():
            ax.set_ylabel('')
            ax.set_xlabel('')
        for ax, col in zip(np.diag(g.axes), df.columns):
            ax.set_title(col, y=0.82, fontsize=10)
        return g

    g = sns.PairGrid(df, diag_sharey=False)
    g.map_lower(scatterfunc) 
    g.map_upper(corrfunc) 
    g.map_diag(histfunc)
    g = make_diag_titles(g, df.columns)
    plt.tight_layout()
    plt.show()

#corrplot(filtered_data.loc[:,'radius_mean':'fractal_dimension_mean'])
#corrplot(filtered_data.loc[:,'radius_se':'fractal_dimension_se'])
'''
https://seaborn.pydata.org/examples/many_pairwise_correlations.html
https://ajh1143.github.io/Corr/
'''
def corrdiag(df):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, vmax=0.3, center=0,
                fmt=".2f", square=True, linewidths=0.5, cbar_kws={'shrink':.5})
    plt.show()

#corrdiag(filtered_data.loc[:,'radius_mean':'fractal_dimension_mean'])
#corrdiag(filtered_data.loc[:,'radius_se':'fractal_dimension_se'])

'''
Let’s investigate the correlation between the features using corr function
for Pearson correlation.
'''
def heatmap(df):
    corr = df.corr()
    sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
    plt.show()

#heatmap(filtered_data.loc[:,'radius_mean':'fractal_dimension_worst'])

def structured_heatmap(df):
    df_pal = sns.husl_palette(df.shape[1], s=.45)
    df_lut = dict(zip(df.columns, df_pal))
    df_colors = pd.Series(df.columns, index=df.columns).map(df_lut)
    sns.clustermap(df.corr(), center=0, cmap="vlag",
                   row_colors=df_colors, col_colors=df_colors,
                   linewidths=.75, figsize=(9, 9))
    plt.show()

#structured_heatmap(filtered_data)

'''
Let's find some of the pairs with the highest correlation coeffs.
https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
'''
def highestcorrpairs(df):
    corr = df.corr().abs()
    sol = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
                 .stack().sort_values(ascending=False))
    return sol[sol>0.95]

print('-'*80)
print(highestcorrpairs(filtered_data.loc[:,'radius_mean':'fractal_dimension_worst']))

'''
Are all the thirty features equally important?
https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/
https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
https://medium.com/@zhang_yang/python-code-examples-of-explained-variance-in-pca-a19cfa73a257
'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def reducefeatures(df, ncomp):
    X = df.values
    X = scale(X)
    pca = PCA(n_components=ncomp).fit(X)
    #plt.figure()
    cummulative_explained_var = np.cumsum(pca.explained_variance_ratio_)
    print(cummulative_explained_var)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(cummulative_explained_var)
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Variance (%)') #for each component
    #ax1.set_title('Breast Cancer Dataset')
    features = range(pca.n_components_)
    ax2.bar(features, pca.explained_variance_ratio_, color='black')
    ax2.set_xlabel('PCA features')
    ax2.set_ylabel('variance %')
    ax2.set_xticks(features)
    plt.tight_layout()
    plt.show()

#reducefeatures(filtered_data,30)

'''
https://mclguide.readthedocs.io/en/latest/sklearn/clusterdim.html
http://benalexkeen.com/principle-component-analysis-in-python/
https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2
https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2
'''
filtered_data.drop(filtered_data.columns[[0,1]], axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def pcaScatter(df):
    X_std = StandardScaler().fit_transform(df)
    pca = PCA(n_components=df.shape[1]).fit_transform(X_std)
    df_pca = pd.DataFrame(pca)
    plt.scatter(df_pca[0], df_pca[1], alpha=.1, color='black')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

#pcaScatter(filtered_data)

def getElbow(df, ncomp, ks):
    X_std = StandardScaler().fit_transform(df)
    pca = PCA(n_components=df.shape[1]).fit_transform(X_std)
    df_pca = pd.DataFrame(pca)
    sse = {}
    for k in range(1,ks):
        # Create a KMeans instance with k clusters: model
        kmeans = KMeans(n_clusters=k).fit(df_pca)
        df_pca['clusters'] = kmeans.labels_
        sse[k] = kmeans.inertia_
    plt.plot(list(sse.keys()), list(sse.values()),'-o', color='black' )
    plt.xlabel('Number of clusters, k')
    plt.ylabel('SSE')
    plt.show()

#getElbow(filtered_data,30,10)

'''
Correlation circle plot
https://stackoverflow.com/questions/37815987/plot-a-correlation-circle-in-python
https://github.com/mazieres/analysis/blob/master/analysis.py#L19-34
https://stackoverflow.com/questions/45148539/project-variables-in-pca-plot-in-python
https://gist.github.com/mazieres/10459516
https://github.com/mazieres/analysis/blob/master/analysis.py
https://thehongwudotcom.wordpress.com/2016/02/28/biplot-in-python-optimized-with-color-scatter-plot/
https://stackoverflow.com/questions/22348668/pca-decomposition-with-python-features-relevances
The components_ array has shape (n_components, n_features) so components_[i, j] is already giving you 
the (signed) weights of the contribution of feature j to component i.
If you want to get the indices of the top 3 features contributing to component i 
irrespective of the sign, you can do:
numpy.abs(pca.component_[i]).argsort()[::-1][:3]
'''
from matplotlib.patches import Circle

def circleOfCorrelations(pc_infos, ebouli):
    plt.axis('scaled')
    circle1=plt.Circle((0,0),radius=1, color='g', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    for idx in range(len(pc_infos["PC-0"])):
        x = pc_infos["PC-0"][idx]
        y = pc_infos["PC-1"][idx]
        #plt.plot([0.0,x],[0.0,y],'k-')
        #plt.plot(x, y, 'rx')
        factor = -2.5
        plt.arrow(0, 0, x*factor, y*factor, width=0.005)
        text = plt.annotate(pc_infos.index[idx], xy=(x*factor,y*factor))
        text.set_fontsize(5)
    plt.xlabel("PC-0 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
    plt.ylabel("PC-1 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.title("Circle of Correlations")

def myPCA(df, clusters=None):
    df_norm = StandardScaler().fit_transform(df)
    pca = PCA(n_components=df.shape[1])
    pca_res = pca.fit_transform(df_norm)
    ebouli = pd.Series(pca.explained_variance_ratio_)
    #ebouli.plot(kind='bar', title="Ebouli des valeurs propres")
    #plt.show()
    coef = np.transpose(pca.components_)
    cols = ['PC-'+str(x) for x in range(len(ebouli))]
    pc_infos = pd.DataFrame(coef, columns=cols, index=df.columns)
    circleOfCorrelations(pc_infos, ebouli)
    plt.show()
    #dat = pd.DataFrame(pca_res, columns=cols)
    #if isinstance(clusters, np.ndarray):
    #    for clust in set(clusters):
    #        colors = list("bgrcmyk")
    #        plt.scatter(dat["PC-0"][clusters==clust],dat["PC-1"][clusters==clust],c=colors[clust])
    #else:
    #    plt.scatter(dat["PC-0"],dat["PC-1"])
    #plt.xlabel("PC-0 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
    #plt.ylabel("PC-1 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
    #plt.title("PCA")
    #plt.show()

#myPCA(filtered_data, clusters=None)

'''
https://datascience.stackexchange.com/questions/31700/how-to-print-kmeans-cluster-python
https://stats.stackexchange.com/questions/9850/how-to-plot-data-output-of-clustering
'''

from scipy import cluster
from sklearn.cluster import KMeans

#def getElbow(df, ks):
#    X_std = StandardScaler().fit_transform(df)
#    pca = PCA(n_components=df.shape[1]).fit_transform(X_std)
#    df_pca = pd.DataFrame(pca)
#    initial = [cluster.vq.kmeans(df_pca,i) for i in range(1,ks)]
#    plt.plot([var for (cent,var) in initial])
#    plt.xlabel('Number of clusters, k')
#    plt.ylabel('SSE')

def getElbow(df, ks):
    X_std = StandardScaler().fit_transform(df)
    pca = PCA(n_components=df.shape[1]).fit_transform(X_std)
    df_pca = pd.DataFrame(pca)
    sse = {}
    for k in range(1,ks):
        # Create a KMeans instance with k clusters: model
        kmeans = KMeans(n_clusters=k).fit(df_pca)
        df_pca['clusters'] = kmeans.labels_
        sse[k] = kmeans.inertia_
    plt.plot(list(sse.keys()), list(sse.values()),'-o', color='black' )
    plt.xlabel('Number of clusters, k')
    plt.ylabel('SSE')
    plt.show()
    
#getElbow(filtered_data,10)

from itertools import cycle
def getGroups(df, ncomp, nclust):
    X_std = StandardScaler().fit_transform(df)
    pca = PCA(n_components=ncomp).fit_transform(X_std)
    kmeans = KMeans(n_clusters=nclust)
    labels = kmeans.fit_predict(pca)
    centroids = kmeans.cluster_centers_
    cycol = cycle('bgrcmk')
    for label in range(ncomp):
        plt.scatter(pca[labels==label,0], pca[labels==label,1], color=next(cycol), alpha=0.5)
    plt.show()

getGroups(filtered_data, 6, 6)

'''
https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
https://www.kaggle.com/spidermanxyz/a-cluster-of-colors-principal-component-analysis
'''
