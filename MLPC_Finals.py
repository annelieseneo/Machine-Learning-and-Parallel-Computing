# import pandas
import pandas as pd

# import numpy for working with numbers
import numpy as np
import math

# import plots
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','inline')

# DATA OBSERVATION AND UNDERSTANDING

# DATA SOURCE

# import data from Excel .csv sheet
df = pd.read_csv(r'C:\Users\annel\Downloads\Bike-Sharing-Dataset-day V1.04.csv')

# show first 5 records of dataset
df.head()

# DATA TYPES

# determine object type of dataset
type(df)

# determine attribute data types
df.info()

# NOISES

# no attributes with missing values

# no incomplete records 

# identify impossible and extreme values
# summary statistics of the attributes, including measures of central tendency 
    # and measures of dispersion
df.describe()

# ERRORS

# identify duplicated records
df[df.duplicated(subset = None, keep = False)]

# identify inconsistencies by displaying all unique values of each attribute
for col in df:
    print(df[col].unique())

# ANOMALIES

# identify outliers
df.boxplot(rot = 0, boxprops = dict(color = 'blue'), 
           return_type = 'axes', figsize = (15, 8))
plt.title("Box Plot of Bike Sharing Data", size = 20) # title of plot and its size
plt.suptitle("")
plt.xlabel("Attributes", size = 10) # x axis label
plt.ylabel("Measurements (units)", size = 10) # y axis label
plt.xticks(size = 10) # size of x ticks
plt.yticks(size = 10) # size of y ticks
plt.show()


# smooth outliers for 'hum' using winsorization technique
# replace outlier with maximum or minimum non-outlier 

# compute interquartile range (IQR)
IQR = df['hum'].quantile(0.75) - df['hum'].quantile(0.25)

# compute maximum and minimum non-outlier value
minAllowed = df['hum'].quantile(0.25)-1.5*IQR
maxAllowed = df['hum'].quantile(0.75)+1.5*IQR

# replace outlier values
for i in range(len(df['hum'])): 
    if df['hum'][i] < minAllowed:
       df['hum'] = df['hum'].replace(df['hum'][i], minAllowed)
    elif df['hum'][i] > maxAllowed:
       df['hum'] = df['hum'].replace(df['hum'][i], maxAllowed)
    else: continue

# smooth outliers for 'windspeed' using winsorization technique
# replace outlier with maximum or minimum non-outlier 

# compute interquartile range (IQR)
IQR = df['windspeed'].quantile(0.75) - df['windspeed'].quantile(0.25)

# compute maximum and minimum non-outlier value
minAllowed = df['windspeed'].quantile(0.25)-1.5*IQR
maxAllowed = df['windspeed'].quantile(0.75)+1.5*IQR

# replace outlier values
for i in range(len(df['windspeed'])): 
    if df['windspeed'][i] < minAllowed:
       df['windspeed'] = df['windspeed'].replace(df['windspeed'][i], minAllowed)
    elif df['windspeed'][i] > maxAllowed:
       df['windspeed'] = df['windspeed'].replace(df['windspeed'][i], maxAllowed)
    else: continue

# check that outliers have been smoothed
df.boxplot(rot = 0, boxprops = dict(color = 'blue'), 
           return_type = 'axes', figsize = (15, 8))
plt.title("Box Plot of Bike Sharing Data", size = 20) # title of plot and its size
plt.suptitle("")
plt.xlabel("Attributes", size = 10) # x axis label
plt.ylabel("Measurements (units)", size = 10) # y axis label
plt.xticks(size = 10) # size of x ticks
plt.yticks(size = 10) # size of y ticks
plt.show()

# data reduction
# import searborn library for more variety of data visualisation using fewer 
    # syntax and interesting default themes
import seaborn as sns 

# data reduction: correlation matrix
# compare linear relationships between attributes using correlation 
# coefficient generated using correlation matrix
sns.heatmap(df.corr(), cmap = 'PuBu', annot = True, fmt='.2f')
plt.show()

# REGRESSION

# MULTIPLE LINEAR REGRESSION MODEL

# import train test split module
from sklearn.model_selection import train_test_split

# import linear regression model
import statsmodels.api as sm

# find variance of all attributes
df.var()

df_MLR = df # full dataset

y_MLR = df_MLR.pop('total_rental_bikes ') # target class label

X_MLR = df_MLR # attributes

# train test splits as 80% train set and 20% test set
X_trainMLR, X_testMLR, y_trainMLR, y_testMLR = train_test_split(X_MLR, y_MLR, 
                                                                test_size=0.2, 
                                                                random_state=2222)

# building the model
X_train_MLR = sm.add_constant(X_trainMLR) # adds a constant column to input data 
                                        # set to better consider the y intercept
lr_1 = sm.OLS(y_trainMLR, X_train_MLR).fit() # fit train data to model
lr_1.summary() # summary of OLS Regression Results

# import to check the VIF values of the attributes 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# create a dataframe to list all attributes and their VIFs
vif = pd.DataFrame()
vif['Features'] = X_trainMLR.columns # attributes
vif['VIF'] = [variance_inflation_factor(X_trainMLR.values, i) 
              for i in range(X_trainMLR.shape[1])] # find VIFs
vif['VIF'] = round(vif['VIF'], 2) # VIFs round to 2 decimal places
vif = vif.sort_values(by = "VIF", ascending = False) # sort in descending order
print(vif)

# drop the most statistically insignificant attribute with p-value 
# over 0.05 strongly indicating null hypothesis
X = X_trainMLR.drop('workingday', 1)

# rebuild and fit model to new set of attributes
X_train_MLR = sm.add_constant(X) # adds a constant column to input data set to 
                                # better consider the y intercept
lr_2 = sm.OLS(y_trainMLR, X_train_MLR).fit() # fit model

# summary of OLS Regression Results
lr_2.summary()

# create another dataframe to list new set of attributes and their VIFs
vif = pd.DataFrame() # create dataframe
vif['Features'] = X.columns # new set of attributes
vif['VIF'] = [variance_inflation_factor(X.values, i) 
              for i in range(X.shape[1])] # find VIFs
vif['VIF'] = round(vif['VIF'], 2) # round VIFs to 2 decimal places
vif = vif.sort_values(by = "VIF", ascending = False) # sort in descending order
print(vif)

# drop the most statistically insignificant attribute with p-value 
# over 0.05 strongly indicating null hypothesis
X = X_trainMLR.drop(['workingday','holiday'], 1)

# rebuild and fit model to new set of attributes
X_train_MLR = sm.add_constant(X) # adds a constant column to input data set to 
                                # better consider the y intercept
lr_3 = sm.OLS(y_trainMLR, X_train_MLR).fit() # fit model

# summary of OLS Regression Results
lr_3.summary()

# create another dataframe to list new set of attributes and their VIFs
vif = pd.DataFrame() # create dataframe
vif['Features'] = X.columns # new set of attributes
vif['VIF'] = [variance_inflation_factor(X.values, i) 
              for i in range(X.shape[1])] # find VIFs
vif['VIF'] = round(vif['VIF'], 2) # round VIFs to 2 decimal places
vif = vif.sort_values(by = "VIF", ascending = False) # sort in descending order
print(vif)

# drop the attribute with most multicollinearity attribute with VIF over 10
X = X_trainMLR.drop(['workingday','holiday', 'hum'], 1)


# rebuild and fit model to new set of attributes
X_train_MLR = sm.add_constant(X) # adds a constant column to input data set to 
                                # better consider the y intercept
lr_4 = sm.OLS(y_trainMLR, X_train_MLR).fit() # fit model

# summary of OLS Regression Results
lr_4.summary()

# create another dataframe to list new set of attributes and their VIFs
vif = pd.DataFrame() # create dataframe
vif['Features'] = X.columns # new set of attributes
vif['VIF'] = [variance_inflation_factor(X.values, i) 
              for i in range(X.shape[1])] # find VIFs
vif['VIF'] = round(vif['VIF'], 2) # round VIFs to 2 decimal places
vif = vif.sort_values(by = "VIF", ascending = False) # sort in descending order
print(vif)

# residual analysis of train data
y_train_pred = lr_4.predict(X_train_MLR)

# plot histogram of error terms
fig = plt.figure()
sns.distplot((y_trainMLR - y_train_pred), bins = 20) # histogram with 20 bins
fig.suptitle('Error Terms', fontsize = 20) # title of plot 
plt.xlabel('Errors (in units)', fontsize = 18) # x axis label
plt.ylabel('Density', fontsize = 18)
plt.xticks(size = 13) # size of x ticks
plt.yticks(size = 13) # size of y ticks

# predict the test set

# adds a constant column to test set to better consider the y intercept
X_test_m4 = sm.add_constant(X_testMLR)

# drop all attributes that were previously decided as statistically 
# insignificant with p-value over 0.05 strongly indicating null hypothesis
X_test_m4 = X_test_m4.drop(['workingday','holiday', 'hum'], axis = 1)

# make predictions using final model
y_pred_m4 = lr_4.predict(X_test_m4)
y_pred_m5 = lr_4.predict(X_train_MLR)

# find R2 score
from sklearn.metrics import r2_score
r2_score(y_true = y_testMLR, y_pred = y_pred_m4)
r2_score(y_true = y_trainMLR, y_pred = y_pred_m5)

# new data record must be within the data ranges to avoid extrapolation
df.describe()

# create new record
newdata2 = {'season':[1,4],
            'holiday':[0,1],
            'weekday':[0,6],           
            'workingday':[0,1],           
            'weather':[1,4],
            'temp':[0.10,0.20],
            'hum':[0.10,0.70],
            'windspeed':[0.30,0.50]}

# create as dataframe
newdf2 = pd.DataFrame(newdata2, columns=['season','holiday','weekday',
                                         'workingday','weather','temp',
                                         'hum','windspeed'])

# adding constant variable to test dataframe
X_test_new = sm.add_constant(newdf2, has_constant='add')

# create X_test dataframe by dropping feature-selected attributes 
# from the latest X_test_m4
X_test_new = X_test_new.drop(['workingday','holiday', 'hum'], axis = 1)

# make predictions of target class label
y_pred_new = lr_4.predict(X_test_new)
y_pred_new

# CLASSIFICATION

# DATA PREPROCESSING

# minimum and maximum values of total_rental_bikes
df.describe()
min_value = 22
max_value = 8714

# interval width for each total_rental_bikes class (total will have 3 classes)
interval = (max_value - min_value)/3
print(interval)

bikes = df['total_rental_bikes ']

# data reduction
# concept hierarchy generation and segmentation by natural partitioning of 
    # GrowthRate into three higher level concepts : 1 for Low, 2 for Medium, 
    # and 3 for High 
df['bikes_category'] = [1 if min_value - 1 < i < min_value + interval 
                        else 2 if min_value + interval -1 < i < max_value - interval +1 
                        else 3 for i in bikes ]

# drop original column
df.pop('total_rental_bikes ')

# set bikes_category as the target class object
df['bikes_category'] = df.bikes_category.astype(int)
df['bikes_category'] = df.bikes_category.astype(str)
df['bikes_category'] = df.bikes_category.astype(object)

# display the number of entries, the number of names of the column attributes,
    # the data type and digit placings, and the memory space used
df.info()

# data reduction

# visualise pairs plot or scatterplot matrix to identify 
    # weak class-attribute relationship
g = sns.pairplot(df, hue = 'bikes_category', palette = 'PuBu')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot, common_norm=False)

# import counter
from collections import Counter

# count each outcome of 'bikes_category'
count = Counter(df['bikes_category'])
print(count.items())

# DATA MODELLING

# import DT algorithm from DT class
from sklearn.tree import DecisionTreeClassifier

# split dataset into attributes and labels
X = df.iloc[:, :-1].values # the attributes
y = df.iloc[:, 8].values # the labels

# choose appropriate range of training set proportions
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

# plot DT based on entropy information gain, best splitter, and minimum 2
    # sample leaves
DT = DecisionTreeClassifier(splitter = 'best', criterion = 'entropy', 
                            min_samples_leaf = 2)

# find best training set proportion for the chosen DT model
plt.figure()
for s in t:
    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size = 1-s, 
                                                            random_state = 111)
        DT.fit(X_train, y_train) # consider DT scores
        scores.append(DT.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')
plt.title("Accuracies of Training Set Proportions") # title of plot
plt.xlabel('Training Set Proportion') # x axis label
plt.ylabel('Accuracy'); # y axis label

# choose train test splits from original dataset as 80% train data and 
    # 20% test data for highest accuracy
from sklearn.model_selection import train_test_split
X_trainDT, X_testDT, y_trainDT, y_testDT = train_test_split(X, y, 
                                                            test_size=0.2, 
                                                            random_state=111)

# number of records in training set
len(X_trainDT)

# count each outcome in training set
count = Counter(y_trainDT)
print(count.items())

# DT CLASSIFICATION MODEL
# prediction modelling
# using DT classifier based on entropy information gain, best splitter, 
    # and minimum 2 sample leaves
classifierDT = DecisionTreeClassifier(splitter = 'best', criterion='entropy', 
                                      min_samples_leaf = 2)
classifierDT.fit(X_trainDT, y_trainDT)

# import library to plot tree
from sklearn import tree

fig = plt.figure(figsize = (100, 70))
# target class labels
cn = ['1','2','3']
# attribute features names
fn = ['season','holiday','weekday','workingday','weather','temp',
      'hum','windspeed']

# plot DT
DT = tree.plot_tree(classifierDT,
                    feature_names = fn,
                    class_names= cn,
                    filled = True)

# identifies the important features
classifierDT.feature_importances_

# extracted rules
dtrules = tree.export_text(classifierDT, feature_names = fn)
print(dtrules)

# model evaluation
# number of records in test set
len(X_testDT)

# count each outcome in test set
count = Counter(y_testDT)
print(count.items())

# use the chosen DT models to make predictions on test data
y_predDT = classifierDT.predict(X_testDT)

# using confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_testDT, y_predDT))
print(classification_report(y_testDT, y_predDT))

# using accuracy performance metric
from sklearn.metrics import accuracy_score
print("Train Accuracy: ", accuracy_score(y_trainDT, 
                                         classifierDT.predict(X_trainDT)))
print("Test Accuracy: ", accuracy_score(y_testDT, y_predDT))

# new data record must be within the data ranges to avoid extrapolation
df_desc = df.describe()

# create new record
newdata = {'season':[1],
            'holiday':[0],
            'weekday':[0],           
            'workingday':[0],           
            'weather':[1],
            'temp':[0.10],
            'hum':[0.10],
            'windspeed':[0.30]}

# create as dataframe
newdf = pd.DataFrame(newdata, columns=['season','holiday','weekday',
                                       'workingday','weather','temp',
                                       'hum','windspeed'])

# compute probabilities of assigning to each of the two classes of bikes_category
probaDT = classifierDT.predict_proba(newdf)
probaDT.round(4) # round probabilities to four decimal places, if applicable

# make prediction of class label
predDT = classifierDT.predict(newdf)
predDT