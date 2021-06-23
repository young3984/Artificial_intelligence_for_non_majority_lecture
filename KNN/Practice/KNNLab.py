# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 00:16:35 2020

@author: sin18
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier

#pip install scikit-learn=0.21
iris=load_iris()

# load 다른게 하는 방법 #

# from sklearn import datasets()
# iris =datasets.load_iris()
# dia=datasets.load_diabetes()

#######################

print(iris.data)        # 데이터만 찍힌다
print(iris.target)      # class 번호만 찍힌다
print(iris.feature_names)   # 이름을 알려준다

# 학습데이터와 테스트 데이터 나눠야 된다.  

X=iris.data
Y=iris.target

print(X)
print(Y)

X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2) 

print(X_Test.shape)
print(X_Train.shape)
print(Y_Test.shape)
print(Y_Train.shape) 

knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_Train,Y_Train)

Y_Pred=knn.predict_proba(X_Test)

print(Y_Pred)       # 테측한 것 
print(Y_Test)       # 실제 값





















