# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:07:13 2020

@author: anmol
"""
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
df = pd.read_excel(r'Sparks_grips_Data.xlsx')
print(df['Scores'])
X = np.array(df['Hours']).reshape(-1, 1) 
y = np.array(df['Scores']).reshape(-1, 1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
regr = LinearRegression() 
regr.fit(X_train, y_train) 
print(regr.score(X_test, y_test)) 
y_pred = regr.predict(X_test) 
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, y_pred, color ='k')