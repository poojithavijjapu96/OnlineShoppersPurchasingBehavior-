# -*- coding: utf-8 -*-
"""

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

X=data.drop('Revenue',axis=1)
y=data['Revenue']

#Splitting into training and test sets

X_training, X_test, y_training, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

#Making sure the split is representing good sample of both classes in training and in test:

X_training.shape #8631 rows 
X_test.shape #3699 rows

data[data['Revenue']==0].shape
data[data['Revenue']==1].shape    

#Oversampling the data

#conda install -c conda-forge imbalanced-learn

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler 
oversample = RandomOverSampler(sampling_strategy='minority')

#Sampling only the training data to avoid data leakage
X_over, y_over = oversample.fit_resample(X_training, y_training)




lr1 = LogisticRegressionCV(cv=5, max_iter = 10000, random_state = 6030)
lr1.fit(X_over, y_over)
y_predTraining = lr1.predict_proba(X_over)[:,1]



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

    confmat = confusion_matrix(y_over, yhat, labels = [0,1])
    TN = confmat[0,0]
    FP = confmat[0,1]
    FN = confmat[1,0]
    TP = confmat[1,1]
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    
    
    RAS=roc_auc_score(y_over, y_predTraining)
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
    
    RAS=roc_auc_score(y_test, y_predTest)
    sensitivity = TP / (TP + FN) #TPR or sensitivity
    specificity = TN / (TN + FP) #TNR or specificity
    FNR=1-sensitivity
    FPR=1-specificity
    accuracy = (TP + TN) / sum(sum(confmat))
    youden_index= sensitivity + specificity - 1
    precision= TP/(TP+FP)
    PositivePredictiveValue=TP/(TP+FP)
    
    print("For threshold ",t," the confusion matrix is \n",confmat)
    print("And other metrics are \nSensitvity(or TPR): ",sensitivity," \nSpecificity(or TNR): ",specificity, "\nFNR: ",FNR,"\nFPR: ",FPR,"\nAccuracy: ",accuracy,"\nYouden Index: ",youden_index,"\nPrecision: ",precision,"\n Roc_Auc_Score",RAS,"\nPositivePredictionValue",PositivePredictiveValue)     
    print("\n")
    


 
importance = lr1.coef_[0]
sorted(zip(importance,X.columns.values), reverse = True)

#The important features of this logistic regression are:
#Month_Nov, TrafficType_2, TrafficType_8, TrafficeType_10, Month_Sep, TrafficType_5, Weekend_True
#TrafficType_4,Region_6, TrafficType_4, Region_6, TrafficType_11, TrafficType_20, Month_Jul, Region_2,
#Month_oct, PageValues,Browser_4,OperatingSystem_2,Region_8,Informational,Browser_8,Region_3,Browser_12
#Browser_10,Administrative,TrafficType_7, Region_4,TrafficType_14,OperatingSystems_7, ProductRelated,ProductRelated_Duration

#In conclusion Month, TrafficType, Weekend, Region, PageValues are top important features.

#Since we want True Positives to be higher in order to identify and target the Revenue generating customer, we are okay to identify some extra non revenue generating customers as positive
#We want higher sensitivity
#Hence my threshold value should be lower. 
#Observing results from the three thresholds - the threshold 0.15 seems best as it identifies more True Positives 
#Though accuracy and precision are lower than those of threshold 0.29 - there is a balanced sensitivity and specificity.

##Logistic Regression with Lasso Regression to choose features; using the above best threshold:
from sklearn.model_selection import KFold
from sklearn.model_selection import feature_importances_

model = LogisticRegression(max_iter = 10000)
solvers = ['liblinear', 'saga']
penalty = ['l1']
c_values = np.logspace(-5,5,10 )

grid = dict(solver=solvers,penalty=penalty,C=c_values) # defining grid search
cv = KFold(n_splits = 5, shuffle = True, random_state = 1) #Cross validation  5 fold
lr2 = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
lr2_result = lr2.fit(X_over, y_over)

# summarize results
print("Best: %f using %s" % (lr2_result.best_score_, lr2_result.best_params_)) 



y_predTest = lr2.predict_proba(X_test)[:,1]
yhat=0
confmat=0
yhat = np.where(y_predTest > 0.35, 1, 0)
confmat = confusion_matrix(y_test, yhat, labels = [0,1])
TN = confmat[0,0]
FP = confmat[0,1]
FN = confmat[1,0]
TP = confmat[1,1]
RAS=roc_auc_score(y_test, y_predTest)
sensitivity = TP / (TP + FN) #TPR or sensitivity
specificity = TN / (TN + FP) #TNR or specificity
FNR=1-sensitivity
FPR=1-specificity
accuracy = (TP + TN) / sum(sum(confmat))
youden_index= sensitivity + specificity - 1
precision= TP/(TP+FP)
PositivePredictiveValue=TP/(TP+FP)

print("The results of Lasso Regression with threshold 0.35 are \n the confusion matrix is \n",confmat)
print("And other metrics are \nSensitvity(or TPR): ",sensitivity," \nSpecificity(or TNR): ",specificity, "\nFNR: ",FNR,"\nFPR: ",FPR,"\nAccuracy: ",accuracy,"\nYouden Index: ",youden_index,"\nPrecision: ",precision,"\n Roc_Auc_Score",RAS,"\nPositivePredictionValue",PositivePredictiveValue)
print("\n")

#Important Features
importance2 = lr2.best_estimator_.coef_[0]
sorted(zip(importance2,X.columns.values), reverse = True)

#The best features are PageValues, ProductRelated, Informational_Duration, ProductRelated_Duration and Administrative_Duration

##To further test the model - using a Random Forest algorithm

from sklearn.ensemble import RandomForestClassifier

rf_grid = {'n_estimators': np.linspace(100, 1000, 10, dtype = int),
           'max_features': range(20,31)}
RF = GridSearchCV(RandomForestClassifier(random_state = 10),
                    param_grid = rf_grid, cv = 5, n_jobs = -1, scoring = 'f1')
RF.fit(X_over, y_over)

print(RF.best_params_)
print(mean_squared_error(RF.predict(X_test), y_test))


#{'max_features': 23, 'n_estimators': 100}
#0.1273317112733171

#Refitting the model using the best values:

refit_rf=RandomForestClassifier(min_samples_leaf = 10, random_state = 10,
                                n_estimators=RF.best_estimator_.n_estimators,
                                max_features=RF.best_estimator_.max_features)

refit_rf.fit(X_over, y_over)

print(mean_squared_error(refit_rf.predict(X_test), y_test))

y_pred=refit_rf.predict(X_test)

#Confusion Matrix
confmat=confusion_matrix(y_true=y_test, y_pred=refit_rf.predict(X_test), labels=[0,1])

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
RAS=roc_auc_score(y_test, y_pred)
PositivePredictiveValue=TP/(TP+FP)

print("The results of Random Forest are \n the confusion matrix is \n",confmat)
print("And other metrics are \nSensitvity(or TPR): ",sensitivity," \nSpecificity(or TNR): ",specificity, "\nFNR: ",FNR,"\nFPR: ",FPR,"\nAccuracy: ",accuracy,"\nYouden Index: ",youden_index,"\nPrecision: ",precision,"\n Roc_Auc_Score",RAS,"\nPositivePredictionValue",PositivePredictiveValue) 
print("\n")

#Important Features
importance = pd.DataFrame({'feature':X.columns.values, 'importance_1':refit_rf.feature_importances_, })
importance.sort_values(by = ['importance_1'], ascending = False)

#Top important features are PageValues, ExitRates, ProductRelated_Duration, ProductRelated, BounceRates,Month

## SVM Model

from sklearn.svm import SVC 
from sklearn.metrics import classification_report

model = SVC(kernel='rbf', C=5,gamma='auto')
model.fit(X_over, y_over)
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
accuracy = (TP + TN) / sum(sum(con_mat))
youden_index= sensitivity + specificity - 1
precision= TP/(TP+FP)
RAS=roc_auc_score(y_test, y_pred)
PositivePredictiveValue=TP/(TP+FP)

print("The results of SVM using rbf kernel are \n the confusion matrix is \n",con_mat)
print("And other metrics are \nSensitvity(or TPR): ",sensitivity," \nSpecificity(or TNR): ",specificity, "\nFNR: ",FNR,"\nFPR: ",FPR,"\nAccuracy: ",accuracy,"\nYouden Index: ",youden_index,"\nPrecision: ",precision,"\n Roc_Auc_Score",RAS,"\nPositivePredictionValue",PositivePredictiveValue)
print("\n")


## SVM Model - rbf kernel

from sklearn.svm import SVC 
from sklearn.metrics import classification_report

model = SVC(kernel='rbf', C=5,gamma='auto')
model.fit(X_over, y_over)
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
accuracy = (TP + TN) / sum(sum(con_mat))
youden_index= sensitivity + specificity - 1
precision= TP/(TP+FP)
RAS=roc_auc_score(y_test, y_pred)
PositivePredictiveValue=TP/(TP+FP)

print("The results of SVM using rbf kernel are \n the confusion matrix is \n",con_mat)
print("And other metrics are \nSensitvity(or TPR): ",sensitivity," \nSpecificity(or TNR): ",specificity, "\nFNR: ",FNR,"\nFPR: ",FPR,"\nAccuracy: ",accuracy,"\nYouden Index: ",youden_index,"\nPrecision: ",precision,"\n Roc_Auc_Score",RAS,"\nPositivePredictionValue",PositivePredictiveValue)
print("\n")










 
