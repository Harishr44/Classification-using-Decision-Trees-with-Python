# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:45:27 2020

@author: Harish
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
data=pd.read_csv("Company_Data.csv")
max(data['Sales'])
min(data['Sales'])
np.mean(data['Sales'])
labels=["bad","average","good"]
bins=[0,5,10,17]
data['Sales']=pd.cut(data['Sales'],bins=bins,labels=labels)

data.Sales.mode()[0]    
data.isnull().sum()
#sales column has one na value
data.Sales=data.Sales.fillna(data.Sales.mode()[0])
data.isnull().sum()
#zeo na values


train,test = train_test_split(data,test_size = 0.2,random_state=0)
from sklearn import preprocessing
string_columns=["ShelveLoc","Urban","US"]
for i in string_columns:
    number=preprocessing.LabelEncoder()
    train[i]=number.fit_transform(train[i])
    test[i]=number.fit_transform(test[i])
    



train_x=train.iloc[:,1:11]
train_y=train.iloc[:,0]
test_x=test.iloc[:,1:11]
test_y=test.iloc[:,0]

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train_x,train_y)
pred_test=model.predict(test_x)
np.mean(pred_test==test_y)
#65% accuracy

