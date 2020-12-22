# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:34:41 2020

@author: iamre
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pickle
matplotlib.rcParams["figure.figsize"]=(20,10)

df1=pd.read_csv("Real estate.csv")
X=df1.iloc[:,1:-1]
y=df1.iloc[:,7]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

import statsmodels.api as sm
X=np.append(arr=np.ones((414,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5,6]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

pickle.dump(regressor_OLS,open('RealEstate.pkl','wb'))
model = pickle.load(open('RealEstate.pkl','rb'))
print(model.predict([[1,2013,31.7,287.603,6,24.9804]]))


t=range(83)

plt.plot(t,y_test,label="Test Set")
plt.plot(t,y_pred)
t=range(83)

plt.plot(t,y_test,label="Actual value")
plt.plot(t,y_pred,label="Predicted value")
plt.legend()
    
    