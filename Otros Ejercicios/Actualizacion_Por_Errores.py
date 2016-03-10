# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:40:48 2015

@author: fernanda
"""


#cd /Users/fernanda/Dropbox/batmelon/aprendizaje/Aprendizaje/datos

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df2=pd.read_csv("regLin.csv")

X_train, X_test, Y_train, Y_test = train_test_split(df2[['X']],df2['y'],train_size=0.75)

#cosas que escribio el profesor para escalar
#es MUY importante normalizar
scaleX=StandardScaler()
scaleY=StandardScaler()
scaleX.fit(X_train)
X_train=scaleX.transform(X_train)
scaleY.fit(Y_train)
Y_train=scaleY.transform(Y_train)


X_train = np.array(X_train);Y_train = np.array(Y_train)
X_test = np.array(X_test);Y_test = np.array(Y_test)

w=np.array([10.0,10.0])

def salida(w,x):
    w = np.array(w)
    x = np.array(x)
    res = float(w[0]+w[1]*x)
    return(res)
    
def entrena(w,X_train,Y_train,eta=0.000001):
    errores=[]
    #errores_almacenados=[]
    #err_sq=0
    for i in range(len(X_train)):
        errores.append(Y_train[i] - (salida(w,X_train[i]) ))
        w_anterior=np.array([element for element in w])
        w[0] = w[0] + eta * ( Y_train[i] - (salida(w_anterior,X_train[i]) ))
        w[1] = w[1] + eta * ( Y_train[i] - (salida(w_anterior,X_train[i]) ))*X_train[i]
       # for m in range(len(X_train)):
        #    err_sq=err_sq + (Y_train[m] - salida(w,X_train[m]))**2
        #errores_almacenados.append(err_sq)    
    return(w,errores)
    
w,errores=entrena(w,X_train,Y_train,eta=0.1)



scaleX=StandardScaler()
scaleY=StandardScaler()
scaleX.fit(X_train)
X_test=scaleX.transform(X_test)
scaleY.fit(Y_train)
Y_test=scaleY.transform(Y_test)

def errores_test(x,y,w):
    errores_test=[]
    for i in range(len(x)):
        errores_test.append(y[i] - (salida(w,x[i]) ))
    return errores_test

errores_test=errores_test(X_test,Y_test,w) 

plt.plot(range(len(X_train)),errores,color='blue')



plt.plot(range(len(X_test)),errores_test,color='green')
