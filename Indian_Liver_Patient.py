# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.discrete.discrete_model as sm
from sklearn.metrics import roc_curve,auc,confusion_matrix

#load the data
df=pd.read_csv('D:/Data summer training/Data_summer_training/Indian_Liver_Patient.csv')
pd.options.display.max_columns=50
df.head()
df.shape

#Find the missing values 
df.isnull().any().any()
df.isnull().sum(axis=0)
df.isnull().sum(axis=0)/len(df)
df.dropna(axis=0,inplace=True)
df.shape
df.isnull().sum(axis=0)

#model
df['Diabetes']=np.where(df.Diabetes==2,1,0)
df['Gender']=np.where(df.Gender=='Male',1,0)
y=df.Diabetes
x=df.iloc[:,:9]
df.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.3,random_state=0)
model=sm.Logit(y_train,x_train)
result=model.fit()
result.summary2()

#deleting the insignificant row
del x_train['Alkphos']
del x_test['Alkphos']
model1=sm.Logit(y_train,x_train)
result1=model1.fit()
result1.summary2()

#parameters of the model
result1.params

#odds ratio
np.exp(result1.params)

#predicted class of test set
probs=result1.predict(x_test)
probs

#predicted class of test set
y_pred=np.where(probs>0.5,1,0)
y_pred

#overall accuracy of the model
np.mean(y_test==y_pred)

#confusion matrix
mat=confusion_matrix(y_test,y_pred)
confusion_matrix(y_test,y_pred)
sns.heatmap(mat,annot=True,cbar=False,fmt='d')
plt.ylabel('True class')
plt.xlabel('Predicted class')

#roc_auc(train set)
prob1=result1.predict(x_train)
fpr1,tpr1,thresholds1=roc_curve(y_train,prob1)
roc_auc1=auc(fpr1,tpr1)
roc_auc1

#roc_auc(test set)
fpr,tpr,thresholds=roc_curve(y_test,probs)
roc_auc=auc(fpr,tpr)
roc_auc    

#Plot ROC curve
plt.title('ROC curve')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'%roc_auc)
plt.legend(loc='best')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('Sensivity')
plt.xlabel('1-Specification')
plt.show()

#sensitivity,specification at various probability thresholds
pd.options.display.max_rows=200
pd.DataFrame({'Sensitivity':tpr,'1-Specification':fpr,'thresholds':thresholds})


