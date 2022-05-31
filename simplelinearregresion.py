# -*- coding: utf-8 -*-
"""
Created on Mon May 30 19:39:38 2022

@author: ariel
"""

#Regresion Lineal Simple 

#Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values                

#Dividir el dataset en el conjunto de entrenamiento y conjunto de test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Crear modelo de regresion lineal simple con el conjunto test

from sklearn.linear_model import LinearRegression 
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test

y_pred = regression.predict(X_test)

#Visualizar los resultados de entrenamiento

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Sueldo vs Años de experiencia (Conjunto de Entrenamiento)')
plt.xlabel('Años de experiencia')
plt.ylabel('Sueldo (en dolares)')
plt.show()

#Visualizar los resultados de testing

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Sueldo vs Años de experiencia (Conjunto de test)')
plt.xlabel('Años de experiencia')
plt.ylabel('Sueldo (en dolares)')
plt.show()