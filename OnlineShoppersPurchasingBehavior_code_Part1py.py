# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:20:40 2022

@author: pooji
"""


from scipy.io import arff
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor  # Decision Tree Regressor
from sklearn.tree import DecisionTreeClassifier  # Decision Tree Classifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

data = pd.read_csv("online_shoppers_intention.csv")

# EDA

data.shape
# This dataset has 18 columns with 12330 observations


pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', data.shape[0]+1)
pd.set_option('max_columns', None)

data.head()

# Each row of this dataset represents a user's consolidated sessions in a given month.


data.isna().sum()
# There are no null values in the data

data.columns

# 10 numerical and 8 categorical attributes

# Columns Present:
# 1.Administrative
# 2.Administrative_Duration
# 3.Informational
# 4.Informational_Duration
# 5.ProductRelated
# 6.ProductRelated_Duration
# 7.BounceRates
# 8.ExitRates
# 9.PageValues
# 10.SpecialDay
# 11.Month- CategoricalVariable
# 12.OperatingSystem - CategoricalVariable
# 13.Browser - CategoricalVariable
# 14.Region - CategoricalVariable
# 15.TrafficType - CategoricalVariable
# 16.VisitorType - CategoricalVariable
# 17.Weekend - CategoricalVariable
# 18.Revenue - CategoricalVariable


# Checking every categorical variable and  their counts

# Month
data.groupby(['Month']).count().Region[:]
# May has highest visitors followed by Nov and March. Either there are no orders for Jan and April or there is no data

# Operating System
data.groupby(['OperatingSystems']).count().Month[:]
# THere are 8 operating systems.Operating System 2 provides highest visitors. But it could also be that it has highest users- this variable might give skewed outputs

# Browser
data.groupby(['Browser']).count().Month[:]
# There are 13 browsers. Browser 2 provides highest visitors followed by browser 1. Similar to OS it could be result

# Region
data.groupby(['Region']).count().Month[:]
# There are 9 regions and region 1 has the highest visitors

# TrafficType
data.groupby(['TrafficType']).count().Month[:]
# There are 20 traffic types and type 2, type 1, type 3 are the highest

# Visitor Type
data.groupby(['VisitorType']).count().Month[:]
# We can see that the returning visitors are much higher than the new_visitors.

# To further check the if they are generating revenue:
data.groupby(['VisitorType', 'Revenue']).count().Month[:]
# Manual Calculation:
# New_Visitor False Revenue :1272/(1272+422) = 75%
# New_Visitor True Revenue: 25%
# Returning_Visitor False: 9081/(9081+1470)=86%
# Returning_Visitor True: 14%
# Value contributed by returning customers is higher

# Weekend
data.groupby(['Weekend', 'Revenue']).count().Month[:]
# Manual Calculation:
# Weekend False Revenue False: 8053/(8053+1409)=85%
# Weekend False Revenue True: 15%
# Weekend True Revenue False: 2369/(2369+499)=83%
# Weekend True Revenue True: 17%
# There is no major difference between weekday and weekend revenue generation


# Analysis on the numerical variables:

# Bar plot of Administrative Duration and Revenue
fig = plt.figure()
plt.bar(data.Revenue, data.Administrative_Duration, color='maroon',
        width=0.4)

plt.xlabel("Revenue")
plt.ylabel("Duration spent on Administrative Page")
plt.show()

# The data is very unbalanced. So trying EDA using only class 1 (gave revenue) to see for any insights.

data_withClass1 = data[data['Revenue'] == 1]
data_withClass0 = data[data['Revenue'] == 0]

data_withClass1.shape
data_withClass0.shape


data_withClass1.groupby(['Month']).count().Region[:]
# November has the highest sales - more than double the next highest sales

data_withClass1.groupby(['Region']).count().Month[:]
# Region 1 has the highest sales-much more higher than others

data_withClass1.groupby(['Region']).count().Month[:]/(data_withClass1.groupby(
    ['Region']).count().Month[:]+data_withClass0.groupby(['Region']).count().Month[:])
# Considering region alone is not giving any meaningful insights on it's impact to generate Revenue


plt.scatter(data.ProductRelated_Duration, data.Revenue, c="orange")

plt.xlabel("Duration spent on Product Related Page")
plt.ylabel("Revenue")
plt.show()


# Duration alone is not giving meaningful insights either.

# Data Manupulation
# Converting all the categorical variables to dummy

MonthDummy = pd.get_dummies(data.Month, prefix='Month')
del MonthDummy['Month_Aug']

OperatingSystemsDummy = pd.get_dummies(data.OperatingSystems, prefix='OperatingSystems')
del OperatingSystemsDummy['OperatingSystems_1']

BrowserDummy = pd.get_dummies(data.Browser, prefix='Browser')
del BrowserDummy['Browser_1']

RegionDummy = pd.get_dummies(data.Region, prefix='Region')
del RegionDummy['Region_1']

TrafficTypeDummy = pd.get_dummies(data.TrafficType, prefix='TrafficType')
del TrafficTypeDummy['TrafficType_1']

VisitorTypeDummy = pd.get_dummies(data.VisitorType, prefix='VisitorType')
del VisitorTypeDummy['VisitorType_New_Visitor']

WeekendDummy = pd.get_dummies(data.Weekend, prefix='Weekend')
del WeekendDummy['Weekend_False']


RevenueDummy=pd.get_dummies(data.Revenue,prefix='Revenue')
del RevenueDummy['Revenue_False']

data=pd.concat([data,MonthDummy,OperatingSystemsDummy,BrowserDummy,RegionDummy,TrafficTypeDummy,VisitorTypeDummy,WeekendDummy,RevenueDummy],axis=1)

del data['Month'] 
del data['OperatingSystems']
del data['Browser']
del data['Region']
del data['TrafficType']
del data['VisitorType']
del data['Weekend']
del data['Revenue']

data.head()

data.rename(columns={'Revenue_True':'Revenue'},inplace=True)

data.shape
#Now there are 12330 rows and 69 columns
data.isna().sum()
#There are no null values

##Data Splitting

#Splitting the response variables and predictors:

X=data.drop('Revenue',axis=1)
y=data['Revenue']

#Splitting into training and test sets

X_training, X_test, y_training, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

#Making sure the split is representing good sample of both classes in training and in test:

X_training.shape #8631 rows 
X_test.shape #3699 rows

data[data['Revenue']==0].shape
data[data['Revenue']==1].shape    

y_training.value_counts()
y_test.value_counts()

#There are 10422 0's and 1908 1's (85% and 15% approx)
#Training has 85% 0's and 15% 1's approximately
#Test has 85% and 15% 1's approximately
#Data division seems okay

##Logistic Regression:

#Starting with logistic regression as the response variable has a binary classification of True(1) or False(0)

#(using sklearn)

#Logistic Regression without any kind of tunning

lr1 = LogisticRegressionCV(cv=5, max_iter = 10000, random_state = 6030)
lr1.fit(X_training, y_training)
y_predTraining = lr1.predict_proba(X_training)[:,1]

#Finding the best threshold for the model:
thresholds=np.linspace(0,1,100)
thresholds=np.delete(thresholds,[99])

#Finding the best values of FPR, accuracy and youden_index and finding the corresponding thresholds
FPR=[]
accuracy=[]
youden_index=[]
ppv=[]
RocAucScore=[]

for a in thresholds:
    yhat = np.where(y_predTraining > a, 1, 0)

    confmat = confusion_matrix(y_training, yhat, labels = [0,1])
    TN = confmat[0,0]
    FP = confmat[0,1]
    FN = confmat[1,0]
    TP = confmat[1,1]
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    
    RAS=roc_auc_score(y_training, y_predTraining)
    RocAucScore.append(RAS)
    youden_index.append(sensitivity + specificity - 1)
    FPR.append(1-specificity)
    Acc = (TP + TN) / sum(sum(confmat))
    accuracy.append(Acc)
    PositivePredictiveValue=TP/(TP+FP)
    ppv.append(PositivePredictiveValue)
    
        

t1=thresholds[np.argmin(FPR)]
t2=thresholds[np.argmax(accuracy)]
t3=thresholds[np.argmax(youden_index)]
t4=thresholds[np.argmax(ppv)]
t5=thresholds[np.argmax(RocAucScore)]


bestthresholds=[t1,t2,t3,t4,t5]

#Choosing the best threshold by checking the model on test data in the above:

y_predTest = lr1.predict_proba(X_test)[:,1]


for t in bestthresholds:
    yhat=0
    confmat=0
    yhat = np.where(y_predTest > t, 1, 0)
    confmat = confusion_matrix(y_test, yhat, labels = [0,1])
    TN = confmat[0,0]
    FP = confmat[0,1]
    FN = confmat[1,0]
    TP = confmat[1,1]
    
    
    sensitivity = TP / (TP + FN) #TPR or sensitivity
    specificity = TN / (TN + FP) #TNR or specificity
    FNR=1-sensitivity
    FPR=1-specificity
    accuracy = (TP + TN) / sum(sum(confmat))
    youden_index= sensitivity + specificity - 1
    precision= TP/(TP+FP)
    RAS=roc_auc_score(y_test, y_predTest)
    PositivePredictiveValue=TP/(TP+FP)
    
    print("For threshold ",t," the confusion matrix is \n",confmat)
    print("And other metrics are \nSensitvity(or TPR): ",sensitivity," \nSpecificity(or TNR): ",specificity, "\nFNR: ",FNR,"\nFPR: ",FPR,"\nAccuracy: ",accuracy,"\nYouden Index: ",youden_index,"\nPrecision: ",precision,"\n Roc_Auc_Score",RAS,"\nPositivePredictionValue",PositivePredictiveValue)     
    print("\n")
    

#Since we want True Positives to be higher in order to identify and target the Revenue generating customer, we are okay to identify some extra non revenue generating customers as positive
#We want higher sensitivity
#Hence my threshold value should be lower. 
#Observing results from the three thresholds - the threshold 0.15 seems best as it identifies more True Positives 
#Though accuracy and precision are lower than those of threshold 0.29 - there is a balanced sensitivity and specificity.

##Logistic Regression with Lasso Regression to choose features; using the above features:
from sklearn.model_selection import KFold


model = LogisticRegression(max_iter = 10000)
solvers = ['liblinear', 'saga']
penalty = ['l1']
c_values = np.logspace(-5,5,10 )

grid = dict(solver=solvers,penalty=penalty,C=c_values) # defining grid search
cv = KFold(n_splits = 5, shuffle = True, random_state = 1) #Cross validation  5 fold
lr2 = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
lr2_result = lr2.fit(X_training, y_training)

# summarize results
print("Best: %f using %s" % (lr2_result.best_score_, lr2_result.best_params_)) 



y_predTest = lr2.predict_proba(X_test)[:,1]
yhat=0
confmat=0
yhat = np.where(y_predTest > 0.15, 1, 0)
confmat = confusion_matrix(y_test, yhat, labels = [0,1])
TN = confmat[0,0]
FP = confmat[0,1]
FN = confmat[1,0]
TP = confmat[1,1]


sensitivity = TP / (TP + FN) #TPR or sensitivity
specificity = TN / (TN + FP) #TNR or specificity
FNR=1-sensitivity
FPR=1-specificity
accuracy = (TP + TN) / sum(sum(confmat))
youden_index= sensitivity + specificity - 1
precision= TP/(TP+FP)
RAS=roc_auc_score(y_test, y_predTest)
PositivePredictiveValue=TP/(TP+FP)


print("The results of Lasso Regression with threshold 0.15 are \n the confusion matrix is \n",confmat)
print("And other metrics are \nSensitvity(or TPR): ",sensitivity," \nSpecificity(or TNR): ",specificity, "\nFNR: ",FNR,"\nFPR: ",FPR,"\nAccuracy: ",accuracy,"\nYouden Index: ",youden_index,"\nPrecision: ",precision)     
print("\n")

#The results of Lasso Regression with threshold 0.15 are 
# the confusion matrix is 
# [[2572  577]
# [ 105  445]]
#And other metrics are 
#Sensitvity(or TPR):  0.8167672276913306  
#Specificity(or TNR):  0.8090909090909091 
#FNR:  0.1832327723086694 
#FPR:  0.19090909090909092 
#Accuracy:  0.8156258448229251 
#Youden Index:  0.6258581367822398 
#Precision:  0.960776989166978

#The sensitivity is 81.6%. Much better than without tunning and penalizaing.

##To further test the model - using a Random Forest algorithm

from sklearn.ensemble import RandomForestClassifier

rf_grid = {'n_estimators': np.linspace(100, 1000, 10, dtype = int),
           'max_features': range(20,31)}
RF = GridSearchCV(RandomForestClassifier(min_samples_leaf = 10, random_state = 10),
                    param_grid = rf_grid, cv = 5, n_jobs = -1, scoring = 'f1')
RF.fit(X_training, y_training)

print(RF.best_params_)
print(mean_squared_error(RF.predict(X_test), y_test))


#{'max_features': 28, 'n_estimators': 400}
#0.09732360097323602

#Refitting the model using the best values:

refit_rf=RandomForestClassifier(min_samples_leaf = 10, random_state = 10,
                                n_estimators=RF.best_estimator_.n_estimators,
                                max_features=RF.best_estimator_.max_features)

refit_rf.fit(X_training, y_training)

print(mean_squared_error(refit_rf.predict(X_test), y_test))
#0.09732360097323602

#Confusion Matrix
confmat=confusion_matrix(y_true=y_test, y_pred=refit_rf.predict(X_test), labels=[0,1])
#array([[3019,  130],
#       [ 230,  320]], dtype=int64)

TN = confmat[0,0]
FP = confmat[0,1]
FN = confmat[1,0]
TP = confmat[1,1]


sensitivity = TP / (TP + FN) #TPR or sensitivity
specificity = TN / (TN + FP) #TNR or specificity
FNR=1-sensitivity
FPR=1-specificity
accuracy = (TP + TN) / sum(sum(confmat))
youden_index= sensitivity + specificity - 1
precision= TP/(TP+FP)
RAS=roc_auc_score(y_test, y_predTest)
PositivePredictiveValue=TP/(TP+FP)

print("The results of Random Forest are \n the confusion matrix is \n",confmat)
print("And other metrics are \nSensitvity(or TPR): ",sensitivity," \nSpecificity(or TNR): ",specificity, "\nFNR: ",FNR,"\nFPR: ",FPR,"\nAccuracy: ",accuracy,"\nYouden Index: ",youden_index,"\nPrecision: ",precision,"\n Roc_Auc_Score",RAS,"\nPositivePredictionValue",PositivePredictiveValue)
print("\n")




#Important Features
importance = pd.DataFrame({'feature':X.columns.values, 'importance_1':refit_rf.feature_importances_, })
importance.sort_values(by = ['importance_1'], ascending = False)


## SVM Model

from sklearn.svm import SVC 
from sklearn.metrics import classification_report

model = SVC(kernel='rbf', C=5,gamma='auto')
model.fit(X_training, y_training)
acc = model.score(X_test,y_test)
print(model.score(X_test,y_test))


y_pred=model.predict(X_test)

print(classification_report(y_test, y_pred))

con_mat=confusion_matrix(y_test,y_pred)
TN = con_mat[0,0]
FP = con_mat[0,1]
FN = con_mat[1,0]
TP = con_mat[1,1]

sensitivity = TP / (TP + FN) #TPR or sensitivity
specificity = TN / (TN + FP) #TNR or specificity
FNR=1-sensitivity
FPR=1-specificity
accuracy = (TP + TN) / sum(sum(confmat))
youden_index= sensitivity + specificity - 1
precision= TP/(TP+FP)
RAS=roc_auc_score(y_test, y_predTest)
PositivePredictiveValue=TP/(TP+FP)

print("The results of SVM using rbf kernel are \n the confusion matrix is \n",confmat)
print("And other metrics are \nSensitvity(or TPR): ",sensitivity," \nSpecificity(or TNR): ",specificity, "\nFNR: ",FNR,"\nFPR: ",FPR,"\nAccuracy: ",accuracy,"\nYouden Index: ",youden_index,"\nPrecision: ",precision,"\n Roc_Auc_Score",RAS,"\nPositivePredictionValue",PositivePredictiveValue)
print("\n")







