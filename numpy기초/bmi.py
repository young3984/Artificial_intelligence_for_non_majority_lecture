# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:29:28 2020

@author: sin18
"""

import numpy as nu

heights=[1.83, 1.76, 1.69, 1.86, 1.77,1.73]
weights=[86, 74, 59, 95, 80, 68]

np_heights=nu.array(heights)
np_weights=nu.array(weights)

bmi=np_weights/(np_heights**2)
print(bmi) 