# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:34:17 2020

@author: sin18
"""
'''
import matplotlib.pylab as plt
from sklearn import linear_model

reg=linear_model.LinearRegression()

X=[[0],[1],[2]]
y=[3, 3.5,5.5]

reg.fit(X,y)

plt.scatter(X,y,color='black')

y_pred=reg.predict(X)

plt.plot(X,y_pred,color='blue',linewidth=3)
plt.show()
'''

import matplotlib.pylab as plt
from sklearn import linear_model

reg = linear_model.LinearRegression()

annual_salary = [[6853], [6889], [7012], [7036], [7116], [7150], [7155], [7283], [7332], [7394], [7403], [8040], [8224], [8268], [8860], [9183], [9278], [10539], [10652], [10988]]
satisfaction = [70, 93, 90, 95, 92, 82, 24, 86, 91, 94, 89, 96, 64, 90, 89, 86, 95, 96, 86, 75]

reg.fit(annual_salary,satisfaction)

plt.scatter(annual_salary, satisfaction, color='black')
y_pred = reg.predict(annual_salary)

plt.plot(annual_salary, y_pred, color='blue', linewidth=3)
plt.show()