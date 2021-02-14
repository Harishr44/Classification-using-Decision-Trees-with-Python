# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 18:31:00 2020

@author: Harish
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
data=pd.read_csv("Fraud_check.csv")

data['Taxable.Income'].mean()

max(data['Taxable.Income'])#100000
min(data['Taxable.Income'])#10000
labels=["risky","good"]
bins=[0,30000,100000]
data['Taxable.Income']=pd.cut(data['Taxable.Income'],bins=bins,labels=labels)

data.isnull().sum()
#zero na values
train,test = train_test_split(data,test_size = 0.2,random_state=0)
from sklearn import preprocessing
colnames=list(data.columns)

string_columns=["Undergrad","Marital.Status","Urban"]
for i in string_columns:
    number=preprocessing.LabelEncoder()
    train[i]=number.fit_transform(train[i])
    test[i]=number.fit_transform(test[i])

train_x=train.iloc[:,[0,1,3,4,5]]
train_y=train.iloc[:,2]
test_x=test.iloc[:,[0,1,3,4,5]]
test_y=test.iloc[:,2]

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train_x,train_y)
pred_test=model.predict(test_x)
np.mean(pred_test==test_y)
#67.5




