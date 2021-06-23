# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:54:09 2020

@author: sin18
"""

import matplotlib.pylab as plt
import numpy as np
from sklearn import linear_model as lm

h_reg=lm.LinearRegression()
uh_reg=lm.LinearRegression()

age=[[25],[35],[45],[55],[65]]
age=np.array(age)
h=[52,52,54,58,61]
h=np.array(h)
uh=[49,48,45,40,36]
uh=np.array(uh)

h_reg.fit(age,h)
uh_reg.fit(age,uh)

plt.scatter(age,h,color='yellow')

h_pred=h_reg.predict(age)

plt.plot(age,h_pred,color='black',linewidth=3)
plt.show()

plt.scatter(age,uh,color='blue')

uh_pred=uh_reg.predict(age)

plt.plot(age,uh_pred,color='black',linewidth=3)
plt.show()


error_h=abs(h_pred-h)
error_uh=abs(uh_pred-uh)