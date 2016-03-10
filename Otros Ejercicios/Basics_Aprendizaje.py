# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:12:25 2015

@author: fernanda
"""
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random as random
 # %matplotlib inline

x = np.linspace(10,100,100)
y=[(i + randint(-10,10))**2 for i in x]

X_train, X_test, Y_train, Y_test = train_test_split(x,y,train_size=0.75)

plt.scatter(X_train,Y_train)
plt.plot(x,x**2,color='red')

plt.scatter(X_test,Y_test)
plt.plot(x,x**2,color='red')

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import csv

df=pd.read_csv("personas.csv")
df

gnd=GaussianNB()
#por que diablos necesita array??!
modelo=gnd.fit(df[['altura','peso','pelo']],np.array(df['sexo']))

modelo.predict([1.60,70,20])
#a predict se le puede mandar todo un cacho del df
#ojo que no haya sido usado para entrenar
modelo.predict_proba([1.60,70,20])

#
#
#
#
#ejercicio: regresion lineal
from sklearn.linear_model import LinearRegression
df2=pd.read_csv("regLin.csv")

X_train, X_test, Y_train, Y_test = train_test_split([[i] for i in df2['X']],df2['y'],train_size=0.75)
#tambien sirve esta manera
X_train, X_test, Y_train, Y_test = train_test_split(df2[['X']],df2['y'],train_size=0.75)


lr=linear_model.LinearRegression()
modelo_lineal=lr.fit(X_train,Y_train)

#plot training set and linear regression
plt.scatter(X_train,Y_train,color='purple')
plt.plot(X_train,modelo_lineal.predict(X_train),color='black')

#plot testing set and linear regression
plt.scatter(X_test,Y_test,color='orange')
plt.plot(X_test,modelo_lineal.predict(X_test),color='black')

w0_estimada = float(modelo_lineal.intercept_)
w1_estimada = float(modelo_lineal.coef_)

#
#calculo del error. como cambia?
w0 = np.arange(-5,5,0.01)
w1 = np.arange(-5,5,0.01)

def funcion_error(w0,w1):
    err = []
    for i in range(len(w0)):
        res = sum((Y_train-w0[i]-w1[i]*np.squeeze(X_train))**2)
        err.append(res)
    return(err) 

errores = funcion_error(w0,w1)

plt.plot(w0,errores,color='blue')
plt.plot(w1,errores,color='blue')

