# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:17:58 2015

@author: fernanda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


from sklearn.cross_validation import train_test_split

datos=pd.read_csv('abalone.csv')

#Test tiene 1045 datos
X_train, X_test, Y_train, Y_test = train_test_split(datos[['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']],datos['Sex'],train_size=0.75)

##DECISION TREE
clf = tree.DecisionTreeClassifier()
clf1 = clf.fit(X_train, Y_train)

prediction1=clf1.predict(X_test)

##RANDOM FOREST
clf = RandomForestClassifier(max_depth=4, n_estimators=8, max_features=1)
clf2 = clf.fit(X_train, Y_train)

prediction2=clf2.predict(X_test)


##ADABOOST
clf = AdaBoostClassifier()
clf3 = clf.fit(X_train, Y_train)

prediction3=clf3.predict(X_test)

print('Decision Tree Accuracy = ' + str(1.0*sum(Y_test.as_matrix()==prediction1)/len(Y_test)))
print('Random Forest Accuracy = ' + str(1.0*sum(Y_test.as_matrix()==prediction2)/len(Y_test)))
print('AdaBoost Accuracy = ' + str(1.0*sum(Y_test.as_matrix()==prediction3)/len(Y_test)))

