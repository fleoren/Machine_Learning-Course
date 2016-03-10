# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:49:11 2015

@author: fernanda
"""

#cd /Users/fernanda/Dropbox/batmelon/Aprendizaje\ de\ Máquina


################ MIS FUNCIONES #########

def graph(formula, x_range,col):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y,color=col)  
    #plt.show()
    
#################################    

from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

datos=pd.read_csv('circulo.csv')

plt.scatter(datos['X1'],datos['X2'],c=datos['Y'])
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show()


#convertir a un formato que le guste al paquete
datos_np=np.asarray(datos)

#YA hice el train
X_train, X_test, Y_train, Y_test = train_test_split(datos[['X1','X2']],datos['Y'],train_size=0.75)

datos_x=np.asarray(X_train)
datos_y=np.ravel(np.asarray(Y_train))

#fit del modelo soporte vectorial

#el slack cambia! con más, estrictamente no hay ruido
clf=SVC(C=10,kernel='rbf')
clf.fit(datos_x,datos_y)
print(clf.predict([[0,0]]))


plt.scatter(X_train['X1'],X_train['X2'],marker='o',color=['gray' if i==0 else 'cyan' for i in Y_train])
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],color='green')
plt.show()

#clf.support_vectors_