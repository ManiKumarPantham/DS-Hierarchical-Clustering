'''
    ###############################       PROBLEM STATEMENT       #################################
 Question 2: Perform clustering for the crime data and identify the number of clusters formed and draw inferences. Refer to crime_data.csv dataset.
 Dataset : crime_data.csv
 Objectives :
 Constraint :
    ################################################################################################
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pylab
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

# Reading the Data set into Python
df = pd.read_csv(r'D:/Hands on/08_Hierarchical Clustering - 2/Assignment/crime_data.csv')

#Information of the Dataset
df.info()

# Renaming the Dataset columns
df.rename(columns = {'Unnamed: 0' : 'State'}, inplace = True)

# Print top 5 records
df.head() 

# First moment business decesion / Measure of Central tendency
df.mean()

df.median()

df.mode()

# Second moment business decesion / Measure of Dispersion
df.var()

df.std()

df.iloc[:, 2:].max() - df.iloc[:, 2:].min()

# Third moment business decesion / Measure of Assymetry
df.skew()

# Fourth moment business decesion / Measure of Peakedness
df.kurt()

# Descriptive statistics
df.describe()

# Correlation cofficient
df.corr()

# Calculating sum of null values in the Dataset
df.isnull().sum()

# Identifing the duplicated values
dup = df.duplicated()

# Calculating the duplicating values
dup.sum()

# Multiple boxplots to see if there are any outliers
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6))
plt.subplot_adjust(wspace = 0.75)
plt.show()

# Creating Winsorizer object
rape_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Rape'])

# Fitting and transforming Rape feature values
df['Rape'] = rape_winsor.fit_transform(df[['Rape']])

# Multiple boxplots to see if there are any outliers
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6))
plt.subplot_adjust(wspace = 0.75)
plt.show()

# Normal Q-Q plot
for i in df.columns[1:]:
    stats.probplot(df[i], dist = 'norm', plot = pylab)
    plt.title('Q-Q plot of ' + str(i))
    plt.show()

# Applying YeoJohnsonTransformer on Murder Feature
Murder_transform = YeoJohnsonTransformer('Murder')
df.Murder = Murder_transform.fit_transform(df[['Murder']])
stats.probplot(df.Murder, dist = 'norm', plot = pylab)

# Applying YeoJohnsonTransformer on Assault Feature
Assault_transform = YeoJohnsonTransformer(variables = ['Assault'])
df.Assault = Assault_transform.fit_transform(df[['Assault']])
stats.probplot(df.Assault, dist = 'norm', plot = pylab)

# Applying YeoJohnsonTransformer on UrbanPop Feature
Urbanpop_transform = YeoJohnsonTransformer(variables = ['UrbanPop'])
df.UrbanPop = Urbanpop_transform.fit_transform(df[['UrbanPop']])
stats.probplot(df.UrbanPop, dist = 'norm', plot = pylab)

# Applying YeoJohnsonTransformer on Rape Feature
Rape_transform = YeoJohnsonTransformer(variables = ['Rape'])
df.Rape = Rape_transform.fit_transform(df[['Rape']])
stats.probplot(df.Rape, dist = 'norm', plot = pylab)


# Pairplot
sns.pairplot(df.iloc[:, 1:])

# Correlation cofficient
df.corr()

# Heatmap
sns.heatmap(df.iloc[:, 1:].corr(method = 'pearson'), xticklabels = df.iloc[:, 1:].columns, yticklabels = df.iloc[:, 1:].columns, cmap = 'coolwarm')
plt.title('Heat Map')
plt.tight_layout()
plt.show()

# Gives State unique values
df.State.unique()

# Gives how many times each value repeats
df.State.value_counts()

# Catagorizing numerical data types in one DataFrame
df_num = df.select_dtypes(exclude = 'object')

# Catagorizing Categorical data types in one DataFrame
df_cat = df.select_dtypes(include = 'object')

# Creating a dummy variables to the State column
df_cat1 = pd.get_dummies(df_cat, columns = ['State'], drop_first = True)

# Creating a object of MinMaxScaler
#Murder_scaling = MinMaxScaler()
Murder_scaling = RobustScaler()

# Executing and transforming the feature
df_num1 = pd.DataFrame(Murder_scaling.fit_transform(df_num))

# Statistical values
df_num1.describe()

# Concatinating Numerical and Categorical feaatures
df2 = pd.concat([df_cat1, df_num1], axis = 1)

# Renaming the features
df2.rename({0 : 'Murder', 1 : 'Assult', 2 : 'Urbanpop', 3 : 'Rape'}, inplace = True, axis = 1)

# Statistical values
df2.describe()

# Creating a Dendrogram using ward's method
tree_plot = dendrogram(linkage(df2, method = 'ward'))

'''
shil_score = []
for i in range(2, 10):
    hc1 = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage = 'average')
    y_hc1 = hc1.fit_predict(df2)    
    y_hc1

    df['Cluster_labels'] = y_hc1
    
    shil_score.append(metrics.silhouette_score(df2, y_hc1))
    
print(shil_score)

'''

#Creating a AgglomerativeClustering Object
hc1 = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'average')

# Printing the output labels
y_hc1 = hc1.fit_predict(df2)   

# Printing the output labels 
y_hc1


'''
#Creating a AgglomerativeClustering Object
hc2 = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'complete')

# Printing the output labels
y_hc2 = hc1.fit_predict(df1)

# Printing the output labels
y_hc2

#Creating a AgglomerativeClustering Object
hc3 = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'single')

# Fit and predicting the values
y_hc3 = hc1.fit_predict(df1)

# Printing the output labels
y_hc3

#Creating a AgglomerativeClustering Object
hc4 = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')

# Fit and predicting the values
y_hc4 = hc1.fit_predict(df1)

# Printing the output labels
y_hc4

# Deleting the Cluster_labels column
#df.drop('Cluster_labels', axis = 1, inplace = True)  

'''

# Creating a new feature with label values
df['Cluster_labels'] = pd.Series(y_hc1)

# Ordering the records
df.sort_values(by = 'Cluster_labels', ascending = True)

# Average values of features based on Cluster_lables
df.iloc[:, 1:].groupby(df.Cluster_labels).mean()

# Creating a Dataframe with similar records
cluster0 = df.loc[df.Cluster_labels == 0, :] 
cluster1 = df.loc[df.Cluster_labels == 1, :]

# Accuracy of the model
metrics.silhouette_score(df2, y_hc1) 

# Boxplot
for i in cluster0.columns[1:5]:
    sns.boxplot(x = cluster0[i])
    plt.show()
    

''' 
    ###############################       PROBLEM STATEMENT       #################################
Question 1: Perform clustering for the airlines data to obtain optimum number of clusters. Draw the inferences from the clusters obtained. 
Dataset  : D:/Hands on/08_Hierarchical Clustering - 2/Assignment/EastWestAirlines.

Offerings: For the non-frequent flyers who are more in numbers promotions like more miles per fly, discounted air fare rates can be offered to improve the number of flyers. This offers would help customers to fly frequently. Most of the customers in this cluster did not fly in last 12 months

Objective : 

Constraint : 
    ##############################################################################################
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
from feature_engine.outliers import Winsorizer
from feature_engine import transformation
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import RobustScaler

df = pd.read_excel(r'D:/Hands on/08_Hierarchical Clustering - 2/Assignment/EastWestAirlines.xlsx', sheet_name = 1)

#Information of the Dataset
df.info()

# Print top 5 records
df.head() 

# First moment business decesion / Measure of Central tendency
df.mean()

df.median()

df.mode()

# Second moment business decesion / Measure of Dispersion
df.var()

df.std()

# Third moment business decesion / Measure of Assymetry
df.skew()

# Fourth moment business decesion / Measure of Peakedness
df.kurt()

# Descriptive statistics
df.describe()

# Correlation cofficient
df.corr()

# Calculating sum of null values in the Dataset
df.isnull().sum()

# Identifing the duplicated values
dup = df.duplicated()

# Calculating the duplicating values
dup.sum()

# Columns of df data set
df.columns

# Removing ID# columns since it does not any significance
df.drop('ID#', axis = 1, inplace = True)

# Shape of the data set
df.shape

# for loop to see if there are any outliers
for i in df.columns[0:11]:
    plt.boxplot(df[i])
    plt.title('Boxplot of ' + str(i))
    plt.show()
    

# Creating Winsorizer object and transforming the values of Balance feature
Bal_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Balance'])
df['Balance'] = Bal_winsor.fit_transform(df[['Balance']])

#Qmiles_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Qual_miles'])
#df['Qual_miles'] = Qmiles_winsor.fit_transform(df[['Qual_miles']])

#cc2miles_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['cc2_miles'])
#df['cc2_miles'] = cc2miles_winsor.fit_transform(df[['cc2_miles']])


#cc3miles_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['cc3_miles'])
#df['cc3_miles'] = cc3miles_winsor.fit_transform(df[['cc3_miles']])

# Creating Winsorizer object and transforming the values of Bonus_miles feature
Bmiles_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Bonus_miles'])
df['Bonus_miles'] = Bmiles_winsor.fit_transform(df[['Bonus_miles']])

# Creating Winsorizer object and transforming the values of Bonus_trans feature
Btrans_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Bonus_trans'])
df['Bonus_trans'] = Btrans_winsor.fit_transform(df[['Bonus_trans']])

# Creating Winsorizer object and transforming the values of Flight_miles_12mo feature
fm12mo_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Flight_miles_12mo'])
df['Flight_miles_12mo'] = fm12mo_winsor.fit_transform(df[['Flight_miles_12mo']])

# Creating Winsorizer object and transforming the values of Flight_trans_12 feature
ftrans12_winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, variables = ['Flight_trans_12'])
df['Flight_trans_12'] = ftrans12_winsor.fit_transform(df[['Flight_trans_12']])

# for loop to plot boxplot for all the features
for i in df.columns[0:11]:
    plt.boxplot(df[i])
    plt.title('Boxplot of ' + str(i))
    plt.show()
    
df.var() # Variance
df.corr() # Correlaction coefficiect

# Creating YeoJohnsonTransformer Object
Bal_tf = transformation.YeoJohnsonTransformer(variables = 'Balance')

# Executing and Transforming Balance Feature
df['Balance'] = Bal_tf.fit_transform(df[['Balance']])
stats.probplot(df.Balance, dist = 'norm', plot = pylab)

#dup_Qual_miles = df[['Qual_miles']]

# Gives unique values and its count
df.Qual_miles.value_counts()

# Variance of Qual_miles Data Set
df.Qual_miles.var()


# There are outliers in below features but I want to retain them as is. So, not performing transformation
#Qmiles_tf = transformation.YeoJohnsonTransformer(variables = 'Qual_miles')
#df['Qual_miles'] = Qmiles_tf.fit_transform(df[['Qual_miles']])
#stats.probplot(df.Qual_miles, dist = 'norm', plot = pylab)

#cc1miles_tf = transformation.YeoJohnsonTransformer(variables = 'cc1_miles')
#df['cc1_miles'] = cc1miles_tf.fit_transform(df[['cc1_miles']])
#stats.probplot(df.cc1_miles, dist = 'norm', plot = pylab)

#cc2_miles_tf = transformation.YeoJohnsonTransformer(variables = 'cc2_miles')
#df['cc2_miles'] = cc2_miles_tf.fit_transform(df[['cc2_miles']])
#stats.probplot(df.cc2_miles, dist = 'norm', plot = pylab)

# Applying YeojohnsonTransformation on cc3_miles
#cc3_miles_tf = transformation.YeoJohnsonTransformer(variables = 'cc3_miles')
#df['cc3_miles'] = cc3_miles_tf.fit_transform(df[['cc3_miles']])
#stats.probplot(df.cc3_miles, dist = 'norm', plot = pylab)

# Applying YeojohnsonTransformation on Bonus_miles
Bmiles_tf = transformation.YeoJohnsonTransformer(variables = 'Bonus_miles')
df['Bonus_miles'] = Bmiles_tf.fit_transform(df[['Bonus_miles']])
stats.probplot(df.Bonus_miles, dist = 'norm', plot = pylab)

# Applying YeojohnsonTransformation on Bonus_trans
Btrans_tf = transformation.YeoJohnsonTransformer(variables = 'Bonus_trans')
df['Bonus_trans'] = Btrans_tf.fit_transform(df[['Bonus_trans']])
stats.probplot(df.Bonus_trans, dist = 'norm', plot = pylab)

# Applying YeojohnsonTransformation on Flight_miles_12mo
#fm12mo_tf = transformation.YeoJohnsonTransformer(variables = 'Flight_miles_12mo')
#df['Flight_miles_12mo'] = fm12mo_tf.fit_transform(df[['Flight_miles_12mo']])
#stats.probplot(df.Flight_miles_12mo, dist = 'norm', plot = pylab)

# Applying YeojohnsonTransformation on Days_since_enroll
dse_tf = transformation.YeoJohnsonTransformer(variables = 'Days_since_enroll')
df['Days_since_enroll'] = dse_tf.fit_transform(df[['Days_since_enroll']])
stats.probplot(df.Days_since_enroll, dist = 'norm', plot = pylab)

# Variance of the Data set
df.var() 

# Multiple boxplots to see if there are any outliers
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6))
plt.subplots_adjust(wspace = 1)
plt.show()

# Statistical calclulations
df.describe()

# Columns of the Data set
df.columns

# Shape of the Data set
df.shape

# Creating Robus Sclaing object
scale = RobustScaler()

# Executing and Transforming the Data set features
df2 = pd.DataFrame(scale.fit_transform(df))

# Statistical calclulations
df2.describe()

# Dendrogram using Ward's method
tree_plot = dendrogram(linkage(df2, method = 'ward'))
plt.figure(figsize=(16,7))
plt.show()

'''
######## Tried with Diff values of number of cluster and lilnkage methods ###############

acc_score = []

for i in range(2, 4):
    hc1 = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage = 'average')
    y_hc1 = hc1.fit_predict(df2)       
    y_hc1
    
    df['Cluster_lables'] = pd.DataFrame(y_hc1)
    
    acc_score.append(metrics.silhouette_score(df2, y_hc1))

print(acc_score)
del acc_score

cluster0 = df.loc[df.Cluster_lables == 0, :] 
cluster1 = df.loc[df.Cluster_lables == 1, :]
cluster2 = df.loc[df.Cluster_lables == 2, :]
cluster3 = df.loc[df.Cluster_lables == 3, :]
cluster4 = df.loc[df.Cluster_lables == 4, :]

for i in cluster1.columns[1:5]:
        sns.boxplot(x = cluster1[i])
        plt.show()
    
    df['Cluster_labels'] = y_hc1
'''

# Creating AgglomerativeClustering object
hc1 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'average')

# Executing and predicting the values
y_hc1 = hc1.fit_predict(df2)
y_hc1

# Creating a new feature in df Data set and assigning y_hc1 labels
df['Cluster_lables'] = pd.Series(y_hc1)

# Sorting values based on Cluster_lables feature
df.sort_values(by = 'Cluster_lables', ascending = True, inplace = True)

# Creating a Dataframe with similar records
cluster0 = df.loc[df.Cluster_lables == 0, :] 
cluster1 = df.loc[df.Cluster_lables == 1, :]

# Creating a multiple boxplots
cluster0.plot(kind = 'box', subplots = True, sharey = False, figsize = (10,6))
plt.subplots_adjust(wspace = 0.75)
plt.show()

# Calculating silhourtte score
metrics.silhouette_score(df2, y_hc1)


''' 
    ###############################       PROBLEM STATEMENT       #################################
Question : 3.	Perform clustering analysis on the telecom dataset. The data is a mixture of both categorical and numerical data. It consists of the number of customers who churn. Derive insights and get possible information on factors that may affect the churn decision. Refer to Telco_customer_churn.xlsx dataset.
Dataset  : D:/Hands on/08_Hierarchical Clustering - 2/Assignment/Telco_customer_churn.xlsx.

Objective : 

Constraint : 
    ###############################################################################################
'''
# Importing all required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine.transformation import YeoJohnsonTransformer
import statsmodels.api as sm
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Reading the file into Python
df = pd.read_excel(r'D:/Hands on/08_Hierarchical Clustering - 2/Assignment/Telco_customer_churn.xlsx')

#Information of the Dataset
df.info()

# Print top 5 records
df.head() 

# First moment business decesion / Measure of Central tendency
df.mean()

df.median()

df.mode()

# Second moment business decesion / Measure of Dispersion
df.var()

df.std()

# Third moment business decesion / Measure of Assymetry
df.skew()

# Fourth moment business decesion / Measure of Peakedness
df.kurt()

# Descriptive statistics
df.describe()

# Correlation cofficient
a = df.corr()

# Calculating sum of null values in the Dataset
df.isnull().sum()

# Identifing the duplicated values
dup = df.duplicated()

# Calculating the duplicating values
dup.sum()

# Coulumns of df Data Set
df.columns

# Returns values and its count for Count and Quarter features
df.Count.value_counts()
df.Quarter.value_counts()

# Variance of df_num Data set
df_num.var()

# Total Charge and Total Revenue has corr() = 97.22. So deleting Total Revenue and other insignificant features
df.drop(columns = ['Customer ID', 'Count', 'Quarter', 'Total Revenue', 'Total Refunds'], inplace = True, axis = 1)

# Information of the Data set
df.info()

# Pair plot
sns.pairplot(df)

# Calculating the null values
df.isnull().sum()

# Checking if there are any duplicates values or not
dup = df.duplicated()

# Sum of the duplicates values
dup.sum()

#  Diving the object data types into one Data frame
df_cat = df.select_dtypes(include = 'object')

#  Diving the Non object data types into one Data frame
df_num = df.select_dtypes(exclude = 'object')

# Columns of the df_num Data set
df_num.columns

# Renaming the df_num featues as these features has white spaces
df_num.rename(columns = {'Number of Referrals' : 'Number_of_Referrals', 
                         'Tenure in Months' : 'Tenure_in_Months', 
                         'Avg Monthly Long Distance Charges' : 'Avg_Monthly_Long_Distance_Charges', 
                         'Avg Monthly GB Download' : 'Avg_Monthly_GB_Download', 
                         'Monthly Charge' : 'Monthly_Charge', 
                         'Total Charges' : 'Total_Charges', 
                         'Total Extra Data Charges' : 'Total_Extra_Data_Charges',
                         'Total Long Distance Charges' : 'Total_Long_Distance_Charges'}, inplace = True)

# Infomation of the Data Set
df_num.info()

# Variance of the df_num Data set
df_num.var()

'''
########## Tried different methods to rename the df_num Data set features ############
df_num.columns = df_num.columns.str.replace(' ','_').astype(int)

if df_num.columns.dtype() = 'int':
    df_num.columns = df_num.columns.str.replace(' ','_').astype(int)
else:
    df_num.columns = df_num.columns.str.replace(' ','_').astype(float)
    
'''
# Renaming the df_cat Data set features
df_cat.columns = df_cat.columns.str.replace(' ', '_')
    
# Columns of the df_cat features
df_cat.columns

# Multiple boxplots
df_num.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6))
plt.subplots_adjust(wspace = 0.75)
plt.show()

# for loop to draw boxplots on all the features of df_num
for i in df_num.columns:
    plt.boxplot(df_num[i])
    plt.title('Box plot of ' + str(i))
    plt.show()

# Creating a YeoJohnsonTransformer Object on Avg_Monthly_GB_Download and Transforming its values
gbdown_winsor = YeoJohnsonTransformer('Avg_Monthly_GB_Download')
df_num['Avg_Monthly_GB_Download'] = gbdown_winsor.fit_transform(df_num[['Avg_Monthly_GB_Download']])

# Creating a YeoJohnsonTransformer Object on Total_Long_Distance_Charges and Transforming its values
tldcharges_winsor = YeoJohnsonTransformer('Total_Long_Distance_Charges')
df_num['Total_Long_Distance_Charges'] = tldcharges_winsor.fit_transform(df_num[['Total_Long_Distance_Charges']])


# Boxplot on df_num columns
for i in df_num.columns:
    sns.boxplot(df_num[i])
    plt.title('Box plot of ' + str(i))
    plt.show()

# Normal QQ plot on df_num features
for i in df_num.columns:
    sm.qqplot(df_num[i])
    plt.title('QQ plot of' + str(i))

# Creating Dummy variables for df_cat featues
df_cat = pd.get_dummies(df_cat, columns = df_cat.columns , drop_first = True)

# Creating the RobustScaler object on df_num Data set and transforming its values
robust_scale = RobustScaler()
df_scale = pd.DataFrame(robust_scale.fit_transform(df_num))

# Statistical values
df_scale.describe()

# Concatinating df_cat and df_scale Data sets
df_norm = pd.concat([df_cat, df_scale], axis = 1)

# Dendrogram using ward's method
tree_plot = dendrogram(linkage(df_norm, method = 'ward'))


'''

############# Tried different Cluster values and linkage methods #########
sil_score = []
for i in range(2, 10):
    hc1 = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage = 'single')
    y_hc1 = hc1.fit_predict(df_norm)
    
    sil_score.append(silhouette_score(df_norm, y_hc1))

print(sil_score)
'''

# Creating AgglomerativeClustering object and predicting the labels
hc1 = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'single')

y_hc1 = hc1.fit_predict(df_norm)

y_hc1

# Creating Cluster_labels feature in df Data set and assigining model labels to it
df['Cluster_labels'] = pd.Series(y_hc1)

# Sort the values on Cluster_labels
df.sort_values(by = 'Cluster_labels', ascending = True)

# Creating a new Data frames with respective cluster values
cluster0 = df.loc[df.Cluster_labels == 0]
cluster1 = df.loc[df.Cluster_labels == 1]
cluster2 = df.loc[df.Cluster_labels == 2]
cluster3 = df.loc[df.Cluster_labels == 3]

# Multiple box plot
cluster0.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6))
plt.subplots_adjust(wspace = 0.75)

# Calculating silhouette score
silhouette_score(df_norm, y_hc1)


'''
    ###############################       PROBLEM STATEMENT       #################################
 Question: 4.	Perform clustering on mixed data. Convert the categorical variables to numeric by using dummies or label encoding and perform normalization techniques. The data set consists of details of customers related to their auto insurance. Refer to Autoinsurance.csv dataset.
 
 Dataset : AutoInsurance.csv
 
 Objectives :
 
 Constraint :
    ###############################################################################################
'''

# Importing all the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pylab
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score
from feature_engine. outliers import Winsorizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from feature_engine import transformation

# Read the data into Python
df = pd.read_csv(r"D:/Hands on/08_Hierarchical Clustering - 2/Assignment/AutoInsurance.csv")

# Data types the df Data Set
df.dtypes

# Dropped Unnecessary columns
df.drop(["Customer", "Policy", "Effective To Date"], axis = 1, inplace = True)

# All numerical and categorical variables are seperated
df_num = df.select_dtypes(exclude = ['object'])
df_cat = df.select_dtypes(include = ['object'])

# EDA
df_num.describe()

# 1st moment Business decession
df_num.mean()
df_num.median()
df_num.mode()

# 2nd moment Business decession
df_num.var()
df_num.std()

# 3rd moment Business decession
df_num.skew()

# 4th moment Business decession
df_num.kurt()

# There are no null values in this numerical data set
df_num.info()

# Multiple Box Plot
df_num.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 10))

#Winsorization for treating outliers
win_cust = Winsorizer(capping_method = 'iqr',
                      tail = 'both',
                      fold = 1.5,
                      variables = ['Customer Lifetime Value'])

df_num['Customer Lifetime Value'] = win_cust.fit_transform(df_num[['Customer Lifetime Value']])

win_mpa = Winsorizer(capping_method = 'iqr',
                     tail= 'both',
                     fold = 1.5,
                     variables = ['Monthly Premium Auto'])

df_num['Monthly Premium Auto'] = win_mpa.fit_transform(df_num[['Monthly Premium Auto']])

win_noc = Winsorizer(capping_method = 'iqr',
                     tail = 'both',
                     fold = 1.5,
                     variables = ['Number of Open Complaints'])

df_num['Number of Open Complaints'] = win_noc.fit_transform(df_num[['Number of Open Complaints']])

win_nop = Winsorizer(capping_method = 'iqr',
                     tail = 'both',
                     fold = 1.5,
                     variables = ['Number of Policies'])

df_num['Number of Policies'] = win_nop.fit_transform(df_num[['Number of Policies']])

win_tca = Winsorizer(capping_method = 'iqr',
                     tail = 'both',
                     fold = 1.5,
                     variables = ['Total Claim Amount'])

df_num['Total Claim Amount'] = win_tca.fit_transform(df_num[['Total Claim Amount']])

df_num.plot( kind = 'box', subplots = True, sharey = False, figsize = (10, 10))


#Even after performing winsorization, it is observed that outliers are present
#After observing the column once again, it needs to used as is

#checking Zero Variance
df_num.var()

#Renaming Columns for convenience
df_num = df_num.rename(columns= {'Customer Lifetime Value' : 'Customer_Lifetime_Value',
                                 'Monthly Premium Auto' : 'Monthly_Premium_Auto',
                                 'Number of Open Complaints' : 'Number_of_Open_Complaints',
                                 'Number of Policies' : 'Number_of_Policies',
                                 'Total Claim Amount' : 'Total_Claim_Amount',
                                 'Months Since Last Claim' : 'Months_Since_Last_Claim',
                                 'Months Since Policy Inception' : 'Months_Since_Policy_Inception'})

# Columns of df_num Data set
df_num.columns

# Normal QQ plot on df_num features
stats.probplot(df_num.Income, dist = 'norm', plot = pylab)
stats.probplot(df_num.Customer_Lifetime_Value, dist = 'norm', plot = pylab)
stats.probplot(df_num.Monthly_Premium_Auto, dist = 'norm', plot = pylab)
stats.probplot(df_num.Months_Since_Last_Claim, dist = 'norm', plot = pylab)
stats.probplot(df_num.Number_of_Open_Complaints, dist = 'norm', plot = pylab)
stats.probplot(df_num.Number_of_Policies, dist = 'norm', plot = pylab)
stats.probplot(df_num.Total_Claim_Amount, dist = 'norm', plot = pylab)
stats.probplot(df_num.Months_Since_Last_Claim, dist = 'norm', plot = pylab)


# Creating YeoJohnsonTransformer and transforming df_num values
tf = transformation.YeoJohnsonTransformer()
tf_df_num = tf.fit_transform(df_num)


# Normal QQ plot on df_num features after transformation
stats.probplot(tf_df_num.Income, dist = 'norm', plot = pylab)
stats.probplot(tf_df_num.Customer_Lifetime_Value, dist = 'norm', plot = pylab)
stats.probplot(tf_df_num.Monthly_Premium_Auto, dist = 'norm', plot = pylab)
stats.probplot(tf_df_num.Months_Since_Last_Claim, dist = 'norm', plot = pylab)
stats.probplot(tf_df_num.Number_of_Open_Complaints, dist = 'norm', plot = pylab)
stats.probplot(tf_df_num.Number_of_Policies, dist = 'norm', plot = pylab)
stats.probplot(tf_df_num.Total_Claim_Amount, dist = 'norm', plot = pylab)
stats.probplot(tf_df_num.Months_Since_Last_Claim, dist = 'norm', plot = pylab)

#Normalizaing the data and transform tf_df_num features values 
minmaxscaler = MinMaxScaler()

df_norm = pd.DataFrame(minmaxscaler.fit_transform(tf_df_num))
df_norm.describe()

#Dummy Variable Creation
df_cat = pd.get_dummies(df_cat, drop_first = True)

#df_new1 = tf_df_num.append(df_cat)
#Concatination
df_new = pd.concat([tf_df_num, df_cat], axis = 1)

# Information of df_new Data set
df_new.info()

# Creating Dendrogram using ward's method
tree_plot = dendrogram(linkage(df_new, method = 'ward'))

'''
######### Tried with different cluster values and linkage methods ###################
sil_score = []
for i in range(2, 13):
    y_hc1 = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage = 'average')
    y = y_hc1.fit_predict(df_new)
    sil_score.append(silhouette_score(df_new, y))
    
print(sil_score)
'''

# Creating AgglomerativeClustering object and predicting the cluster labels
hc1 = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'average')
y_hc1 = hc1.fit_predict(df_new)
y_hc1 

# Creating new Cluster_label feature in df Dataset and assigning model labels to it
df['Cluster_label'] = pd.DataFrame(y_hc1)

# Sort the valeus based on Cluster_label
df.sort_values(by = 'Cluster_label', ascending = True)

# Forming diff clusters with respective values
cluster0 = df.loc[df.Cluster_label == 0]
cluster1 = df.loc[df.Cluster_label == 1]

# Calculating the silhouette score
silhouette_score(df_new, y_hc1)

# Multiple boxplot
cluster0.plot(kind = 'box', sharey = False, subplots = True, figsize = (10,6))
plt.subplots_adjust(wspace = 0.75)
    
