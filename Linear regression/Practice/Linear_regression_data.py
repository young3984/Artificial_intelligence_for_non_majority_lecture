# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:26:59 2020

@author: sin18
"""

import matplotlib.pylab as plt
import numpy as np
from sklearn import linear_model as lm
from sklearn import datasets

dia=datasets.load_diabetes()

# print(dia.data)

from sklearn.model_selection import train_test_split
from sklearn import model_selection

# model_selection.train_test_split(dia.data,dia.target,test_size0.2,random_state=0)

X_Train, X_Test,Y_Train,Y_Test=model_selection.train_test_split(dia.data,dia.target,test_size=0.2)

# print(Y_Test)

reg=lm.LinearRegression()
reg.fit(X_Train,Y_Train)

Y_Pred=reg.predict(X_Test)

print(Y_Pred)
print(Y_Test)


plt.plot(Y_Test,Y_Pred,'.') 

x=np.linspace(0,300,100)

y=x
plt.plot(x,y)







