# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:08:36 2015

@author: fernanda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


datos=pd.read_csv('iris.csv')
def func(x):
    if x== 'Iris-setosa':
        return 0
    elif x=='Iris-virginica':
        return 2
    return 1

datos['class_num'] = datos['class'].apply(func)
del datos['class']

#Test tiene 1045 datos
X_train, X_test, Y_train, Y_test = train_test_split(datos[['sepal_length','sepal_width','petal_length','petal_width']],datos['class_num'],train_size=0.99999)

##K MEANS PARA PREPROCESAMIENTO!
clf1 = KMeans(n_clusters=3).fit(X_train)
cluster_pred = clf1.predict(X_train)
#size=len(Y_train)
#Y_pred.reshape(size,1)

plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.scatter(X_train['sepal_length'].as_matrix(), X_train['petal_length'].as_matrix(), c=cluster_pred)
plt.scatter(X_train['sepal_length'].as_matrix(), X_train['petal_length'].as_matrix(), c=Y_train)
plt.title("Classified with K-means")

#SEGMENTACIÓN
X_train['grupos'] = cluster_pred.flatten()
grupo_1 = X_train[X_train.grupos==0].iloc[:,:-1]
grupo_2 = X_train[X_train.grupos==1].iloc[:,:-1]
grupo_3 = X_train[X_train.grupos==2].iloc[:,:-1]

grupo_1_y = Y_train[np.in1d(Y_train.index,grupo_1.index)]
grupo_2_y = Y_train[np.in1d(Y_train.index,grupo_2.index)]
grupo_3_y = Y_train[np.in1d(Y_train.index,grupo_3.index)]


#probar para cada grupo
##DECISION TREE
clf = tree.DecisionTreeClassifier()
clf1 = clf.fit(grupo_1, grupo_1_y)

prediction1=clf1.predict(X_test)

##RANDOM FOREST
clf = RandomForestClassifier(max_depth=4, n_estimators=8, max_features=1)
clf2 = clf.fit(grupo_1, grupo_1_y)

prediction2=clf2.predict(X_test)


##ADABOOST
clf = AdaBoostClassifier()
clf3 = clf.fit(grupo_1, grupo_1_y)

prediction3=clf3.predict(grupo_1)

print('Decision Tree Accuracy Grupo 1= ' + str(1.0*sum(grupo_1_y.as_matrix()==prediction1)/len(grupo_1_y)))
print('Random Forest Accuracy Grupo 1= ' + str(1.0*sum(grupo_1_y.as_matrix()==prediction2)/len(grupo_1_y)))
print('AdaBoost Accuracy Grupo 1 = ' + str(1.0*sum(grupo_1_y.as_matrix()==prediction3)/len(grupo_1_y)))

clf = tree.DecisionTreeClassifier()
clf1 = clf.fit(grupo_2, grupo_2_y)

prediction1=clf1.predict(X_test)

##RANDOM FOREST
clf = RandomForestClassifier(max_depth=4, n_estimators=8, max_features=1)
clf2 = clf.fit(grupo_2, grupo_2_y)

prediction2=clf2.predict(X_test)


##ADABOOST
clf = AdaBoostClassifier()
clf3 = clf.fit(grupo_2, grupo_2_y)

prediction3=clf3.predict(X_test)

print('Decision Tree Accuracy Grupo 2= ' + str(1.0*sum(Y_test.as_matrix()==prediction1)/len(Y_test)))
print('Random Forest Accuracy Grupo 2= ' + str(1.0*sum(Y_test.as_matrix()==prediction2)/len(Y_test)))
print('AdaBoost Accuracy Grupo 2 = ' + str(1.0*sum(Y_test.as_matrix()==prediction3)/len(Y_test)))

clf = tree.DecisionTreeClassifier()
clf1 = clf.fit(grupo_3, grupo_3_y)

prediction1=clf1.predict(X_test)

##RANDOM FOREST
clf = RandomForestClassifier(max_depth=4, n_estimators=8, max_features=1)
clf2 = clf.fit(grupo_3, grupo_3_y)

prediction2=clf2.predict(X_test)


##ADABOOST
clf = AdaBoostClassifier()
clf3 = clf.fit(grupo_3, grupo_3_y)

prediction3=clf3.predict(X_test)

print('Decision Tree Accuracy Grupo 3= ' + str(1.0*sum(Y_test.as_matrix()==prediction1)/len(Y_test)))
print('Random Forest Accuracy Grupo 3= ' + str(1.0*sum(Y_test.as_matrix()==prediction2)/len(Y_test)))
print('AdaBoost Accuracy Grupo 3 = ' + str(1.0*sum(Y_test.as_matrix()==prediction3)/len(Y_test)))



#¿Aumentó alguna métrica segmentando los datos utilizando k-medias como preproceso?