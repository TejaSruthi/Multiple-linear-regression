# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 07:04:21 2019

@author: vhs
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
datset = pd.read_csv('50_Startups.csv')
x = datset.iloc[:, :-1].values
y = datset.iloc[:,-1].values

#encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct  = ColumnTransformer([('State', OneHotEncoder(), [3])],remainder = 'passthrough' )
x = np.array(ct.fit_transform(x).astype('float'))

# avoiding dummy variable trap
x =x[:, 1:]

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state = 0)

# fiiting the multiple regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set result
y_pred = regressor.predict(x_test)

#building the optimal model in backward elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50, 1)), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog= x_opt).fit()
x_opt = x[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()

