#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[26]:


df = pd.read_excel(r'D:\Hands on\08_Hierarchical Clustering - 2\University_Clustering.xlsx')
df


# In[27]:


df.info()


# In[28]:


df.describe()


# In[11]:


import dtale

d = dtale.show(df)

d.open_browser()


# In[17]:


#!pip install AutoClean


# In[29]:


print(df.mean())
print('\n')
print(df.median())
print('\n')
print(df.mode())


# In[31]:


print(df.var())
print('\n')
print(df.std())


# In[32]:


df.skew()


# In[33]:


df.kurt()


# In[34]:


df.describe()


# In[35]:


df.dtypes


# In[37]:


df['UnivID'] = df.UnivID.astype('str')
df.dtypes


# In[39]:


df.drop('UnivID', axis = 1, inplace = True)


# In[40]:


df


# In[42]:


dup = df.duplicated()
sum(dup)
#df.drop_duplicates()


# In[43]:


df.isnull().sum()


# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df.plot(kind = 'box', subplots = True, sharey = False, figsize = (10,6))
plt.subplots_adjust(wspace = 0.75)
plt.show()


# In[58]:


import numpy as np
from sklearn.impute import SimpleImputer
from feature_engine.imputation import RandomSampleImputer

mean_impute = SimpleImputer(missing_values = np.nan, strategy = 'mean')
median_impute = SimpleImputer(missing_values = np.nan, strategy = 'median')
rand_impute = RandomSampleImputer('GradRate')

df['SAT'] = pd.DataFrame(median_impute.fit_transform(df[['SAT']]))
df['SFRatio'] = pd.DataFrame(median_impute.fit_transform(df[['SFRatio']]))
df['GradRate'] = pd.DataFrame(rand_impute.fit_transform(df[['GradRate']]))
df.isnull().sum()


# In[60]:


df.plot(kind = 'box', subplots = True, sharey = False, figsize = (10,6))


# In[73]:


from feature_engine.outliers import Winsorizer

SAT_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['SAT'])
df.SAT = SAT_winsor.fit_transform(df[['SAT']])

Top10_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Top10'])
df.Top10 = Top10_winsor.fit_transform(df[['Top10']])

Accept_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Accept'])
df.Accept = Accept_winsor.fit_transform(df[['Accept']])

SFRatio_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['SFRatio'])
df.SFRatio = SFRatio_winsor.fit_transform(df[['SFRatio']])


# In[76]:


df.plot(kind = 'box', subplots = True, sharey = False, figsize = (10,6))
plt.subplots_adjust(wspace = 0.75)
plt.show()


# In[77]:


df.head()


# In[81]:


df.var()


# In[78]:


df_cat = df.iloc[:, 0:2]
df_cat


# In[80]:


df_num = df.iloc[:, 2:]
df_num


# In[82]:


df_cat.State.unique()


# In[84]:


df_cat.State.value_counts()


# In[93]:


df_cat = pd.get_dummies(df_cat, columns = ['State'], drop_first = True)


# In[94]:


df_cat.shape


# In[89]:


for i in df_num.columns:
    sns.histplot(df_num[i])
    plt.title('histogram of ' + str(i))
    plt.show()


# In[90]:


import scipy.stats as stats
import pylab

for i in df_num.columns:
    stats.probplot(df_num[i], dist = 'norm', plot = pylab)
    plt.title('Normal Q-Q plot of ' + str(i))
    plt.show()
    


# In[96]:


df1 = pd.concat([df_cat, df_num], axis = 1)
df1.head()


# In[97]:


sns.pairplot(df_num)


# In[98]:


df_num.corr()


# In[106]:


matrics = df_num.corr(method = 'pearson')

sns.heatmap(matrics, xticklabels = df_num.columns, yticklabels = df_num.columns, cmap = 'coolwarm')
#plt.tight_layout()
plt.show()
plt.tight_layout()


# In[113]:


def norm_func(i):
    return i - i.min() / (i.max() - i.min())

df_norm = norm_func(df1.iloc[:, 1:])
df_norm.describe()


# In[132]:


from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

tree_plot = dendrogram(linkage(df_norm, method = 'ward'))
plt.figure(figsize = (16, 15))


# In[140]:


hc1 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')

y_hc1 = hc1.fit_predict(df_norm)

y_hc1


# In[146]:


hc2 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'single')

y_hc2 = hc2.fit_predict(df_norm)

y_hc2


# In[148]:


hc3 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'average')

y_hc3 = hc3.fit_predict(df_norm)

y_hc3


# In[151]:


df['Cluster_labels'] = hc1.labels_
df


# In[152]:


df.sort_values(by = 'Cluster_labels', ascending = True)


# In[154]:


df.iloc[:, 1:].groupby(df.Cluster_labels).mean()


# In[157]:


cluster0 = df.loc[df.Cluster_labels == 0, :]
cluster0


# In[159]:


cluster1 = df.loc[df.Cluster_labels == 1, :]
cluster1


# In[161]:


cluster2 = df.loc[df.Cluster_labels == 2, :]
cluster2


# In[163]:


cluster0.to_csv('cluster0.csv', encoding = 'latin1')
cluster1.to_csv('cluster1.csv', encoding = 'latin1')
cluster2.to_csv('cluster2.csv', encoding = 'latin1')

import os
os.getcwd()


# In[173]:


for i in cluster0.columns[2:]:
    sns.boxplot(x = cluster0[i])
    plt.show()

