# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:54:47 2015

@author: fernanda
"""

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################ MIS FUNCIONES #########

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y)  
    plt.show()
    
#################################    

df2=pd.read_csv("regLin4.csv")

X_train, X_test, Y_train, Y_test = train_test_split(df2[['X']],df2['y'],train_size=0.75)

X_train = np.array(X_train);Y_train = np.array(Y_train)
X_test = np.array(X_test);Y_test = np.array(Y_test)

scaleX=StandardScaler()
scaleY=StandardScaler()
scaleX.fit(X_train)
X_train=scaleX.transform(X_train)
scaleY.fit(Y_train)
Y_train=scaleY.transform(Y_train)

plt.scatter(X_train,Y_train)

linReg=LinearRegression()
linReg.fit(X_train,Y_train)

w=np.array([1.0,1.0])

def transferencia(w,x):
    w = np.array(w)
    x = np.array(x)
    if (float(w[0]+w[1]*x)>0):
        res=1
    else:
        res=-1    
    return(res)
    
def entrena(w,X_train,Y_train,eta=0.00001):
    #errores=[]
    for i in range(len(X_train)):
        #errores.append(Y_train[i] - (salida(w,X_train[i]) ))
        w_anterior=np.array([element for element in w])
        w[0] = w[0] + eta * ( Y_train[i] - (transferencia(w_anterior,X_train[i]) ))
        w[1] = w[1] + eta * ( Y_train[i] - (transferencia(w_anterior,X_train[i]) ))*X_train[i]
       # for m in range(len(X_train)):
        #    err_sq=err_sq + (Y_train[m] - salida(w,X_train[m]))**2
        #errores_almacenados.append(err_sq)    
    return(w)
    
w_estimada=entrena(w,X_train,Y_train,eta=0.001)
lam= -(w_estimada[0]/w_estimada[1])


######## programar un AND como perceptron
#######
######
#####
####

def datos(funcion):
    if funcion=='and':
        df3=pd.DataFrame(np.asarray(([0,0,0],[0,1,0],[1,0,0],[1,1,1])),columns=['X1', 'X2', 'y'])
    elif funcion=='or':
        df3=pd.DataFrame(np.asarray(([0,0,0],[0,1,1],[1,0,1],[1,1,1])),columns=['X1', 'X2', 'y'])
    elif funcion=='xor':
        df3=pd.DataFrame(np.asarray(([0,0,0],[0,1,1],[1,0,1],[1,1,0])),columns=['X1', 'X2', 'y'])    
    X_train=df3[['X1','X2']]
    Y_train=df3[['y']]
    Y_train = np.array(Y_train)    
    return(df3,X_train,Y_train)     
    
w=np.array([0.8,-0.4,0.5])

def transferencia_and(w,x):
    w = np.array(w)
    x = np.array(x)
    if (float(w[0]+w[1]*x[0]+w[2]*x[1])>0):
        res=1
    else:
        res=0    
    return(res)
    
def entrena_1(w,X_train,Y_train,eta=0.001):
    w_anterior=np.array([element for element in w])
    he_entrado_al_for=0
    while(he_entrado_al_for==0 or w_anterior!=w or he_entrado_al_for<10000):
        for i in range(len(X_train)):
            #errores.append(Y_train[i] - (salida(w,X_train[i]) ))
            w_anterior=np.array([element for element in w])
            w[0] = w[0] + eta * ( Y_train[i] - (transferencia_and(w_anterior,np.array(X_train.iloc[i])) ))
            w[1] = w[1] + eta * ( Y_train[i] - (transferencia_and(w_anterior,np.array(X_train.iloc[i])) ))*X_train.X1[i]
            w[2] = w[2] + eta * ( Y_train[i] - (transferencia_and(w_anterior,np.array(X_train.iloc[i])) ))*X_train.X2[i]
           #print('nueva')            
            #print(w)
            #print(X_train.iloc[i])
            #print(Y_train[i])
           # for m in range(len(X_train)):
            #    err_sq=err_sq + (Y_train[m] - salida(w,X_train[m]))**2
            #errores_almacenados.append(err_sq)
            he_entrado_al_for=he_entrado_al_for+1
            print(w)
    print('El vector de w es ' + str(w))        
    return(w)

#df3,X_train,Y_train = datos('and')
df3,X_train,Y_train = datos('or')
#df3,X_train,Y_train = datos('xor')

w_estimada=entrena_1(w,X_train,Y_train,eta=0.001)

#plt.scatter(X_train['X1'],X_train['X2'])
