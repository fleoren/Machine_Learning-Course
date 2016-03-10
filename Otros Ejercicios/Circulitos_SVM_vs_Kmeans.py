# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:54:59 2015

@author: fernanda
"""

import numpy as np
import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

n = 5000
nclass = 5
np.random.seed(124755)
X = np.random.rand(n,2)
X = X*20 - 10
centroids = 10*np.random.randn(nclass,2)
y = np.random.choice(range(nclass), n)
X = X + centroids[y]

#mejor idea
#NECESITAS LLENAR TODO EL GRID DE PUNTOS
#Y PONERLE UNO A LOS QUE FUERON GENERADOS ALLA ARRIBA
Y=[]
for point in X:
    i=0
    if np.sqrt((point[0]-centroids[i][0])**2 + (point[1]-centroids[i][1])**2) < 3 or np.sqrt((point[0]-centroids[i+1][0])**2 + (point[1]-centroids[i+1][1])**2) < 3 or np.sqrt((point[0]-centroids[i+2][0])**2 + (point[1]-centroids[i+2][1])**2) < 3 or np.sqrt((point[0]-centroids[i+3][0])**2 + (point[1]-centroids[i+3][1])**2) < 3 or np.sqrt((point[0]-centroids[i+4][0])**2 + (point[1]-centroids[i+4][1])**2) < 3 :
        Y.append(1)
    else:
        Y.append(0)  
#ES UNA SOLA CLASE, SOLAMENTE SEPARADA

plt.pyplot.scatter(X[:,0],X[:,1],c=Y)


#REVISAR CON SVM
#intenté cross validation con valores bajitos y marcaba tooooodo como vectores de soporte
clf1 = SVC(C=10000,kernel='rbf', probability=True)
clf1.fit(X, Y)
prediction1=clf1.predict(X)
plt.pyplot.scatter(X[:,0],X[:,1],c=Y)
plt.pyplot.scatter(clf1.support_vectors_[:,0],clf1.support_vectors_[:,1],color='green')
#REVISAR CON K-NEIGHBORS
clf2 = KNeighborsClassifier(n_neighbors=1)
clf2.fit(X, Y)
prediction2=clf2.predict(X)

print('SVM Accuracy = ' + str(1.0*sum(Y==prediction1)/len(Y)))
print('K Neighbors = ' + str(1.0*sum(Y==prediction2)/len(Y)))




