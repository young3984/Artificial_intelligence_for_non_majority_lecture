# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:54:36 2020

@author: sin18
"""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
import numpy as np

xx = np.arange(0,15)
# xpo = np.arange(0,5)
# xap = np.arange(5,10)
# xpe = np.arange(10,15)


pom=[[70.84,0],[66.24,0],[59.65,0],[61.63,0],[57.81,0]]
ap=[[44.43,0],[43.21,0],[24.90,0],[27.18,0],[25.91,0]]
pea=[[14.37,0],[17.85,0],[11.99,0],[12.29,0],[16.69,0]]
# plt.scatter(pom)
# plt.scatter(ap)
# plt.scatter(pea)


plt.title("sugar content by fruit", fontsize = 12)
plt.xlabel("Fruit number", fontsize = 12)
plt.ylabel("sugar content", fontsize = 12)


X=[[70.84],[66.24],[59.65],[61.63],[57.81],[44.43],[43.21],[24.90],[27.18],[25.91],[14.37],[17.85],[11.99],[12.29],[16.69]]
Y=[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]



plt.scatter(xx,X,color='deepskyblue')
plt.title("sugar content by fruit", fontsize = 12)
plt.xlabel("Fruit number", fontsize = 12)
plt.ylabel("sugar content", fontsize = 12)

X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2) 

knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_Train,Y_Train)

Y_Pred=knn.predict([[40]])
predic=knn.predict(X)

print("에측 : ",Y_Pred)       # 예측한 것 
# print("실제 : ",Y_Test)       # 실제 값

# scores=accuracy_score(Y_Test,Y_Pred)
# print("score : ",scores)       # 정확도 측정

# print(Y)
# print(predic)

