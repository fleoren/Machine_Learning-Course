# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:24:51 2015

@author: fernanda
"""
#cd /Users/fernanda/Dropbox/batmelon/aprendizaje/Aprendizaje/datos


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df3=pd.read_csv("regLinPoli.csv")

#dividir los datos en conjunto para entrenar y para ver si el modelo funciona
X_train, X_test, Y_train, Y_test = train_test_split(df3[['X','X2','X3','X4']],df3['y'],train_size=0.85)

#escalar los datos, escala normal estandar
scaleX=StandardScaler()
scaleY=StandardScaler()
scaleX.fit(X_train)
X_train=scaleX.transform(X_train)
scaleY.fit(Y_train)
Y_train=scaleY.transform(Y_train)


X_train = np.array(X_train);Y_train = np.array(Y_train)
X_test = np.array(X_test);Y_test = np.array(Y_test)

w=np.array([10.0,10.0,10.0,10.0,10.0])

def salida(w,x):
    w = np.array(w)
    x = np.array(x)
    x=np.insert(x,0,1)
    res = float(np.dot(x,w))
    return(res)
    
def entrena(w,X_train,Y_train,eta=0.000001,lam=0.5):
    errores=[]
    errores_almacenados=[]
    for i in range(len(X_train)):
        err_sq=0
        errores.append(Y_train[i] - (salida(w,X_train[i]) ))
        w_anterior=np.array([element for element in w])
        err=Y_train[i] - (salida(w_anterior,X_train[i]) )
        for l in range(len(w)):
            if l==1:
                w[l] = w[l] + eta * err
            elif l==1>1:
                w[l] = w[l] + eta * err*X_train[i-1] - lam*w[l]
        for m in range(len(X_train)):
            err_sq = err_sq + (Y_train[m] - salida(w,X_train[m]))**2    
        errores_almacenados.append(err_sq)  
    return(errores,errores_almacenados,w)
    
errores,errores_almacenados,w=entrena(w,X_train,Y_train,eta=0.0000001,lam=0.2)    
plt.plot(range(len(X_train)),errores_almacenados,color='purple')

#escalar los datos de prueba. nota: se escala con los datos de escalamiento
#de los de entrenamiento
scaleX=StandardScaler()
scaleY=StandardScaler()
scaleX.fit(X_train)
X_test=scaleX.transform(X_test)
scaleY.fit(Y_train)
Y_test=scaleY.transform(Y_test)

def lambda_costo_test(w,x,y):
    lam=np.linspace(0,0.010,num=10)
    error=[]
    for i in range(len(lam)):
        err_sq=0
        err,err_al,w=entrena(w,x,y,eta=0.000001,lam=lam[i])
        for m in range(len(X_train)):
            err_sq = err_sq + (y[m] - salida(w,x[m]))**2
        error.append(err_sq)  
    return lam,error
    

#plt.plot(range(len(X_train)),errores,color='blue')


#plt.plot(range(len(X_test)),errores_test,color='green')


