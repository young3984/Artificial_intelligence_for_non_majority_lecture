# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:10:33 2020

@author: sin18
"""

import matplotlib.pylab as plt
from sklearn import linear_model
 
reg=linear_model.LinearRegression()
 
X=[[174],[152],[138],[128],[186]]
y=[[71],[55],[46],[38],[88]]
 
reg.fit(X,y)
 
plt.scatter(X,y,color='black')

y_pred=reg.predict(X)

plt.plot(X,y_pred,color='blue',linewidth=3)
plt.show() 