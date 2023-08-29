# -*- coding: utf-8 -*-


##Importar base de datos

#La base de datos que se eligió para realizar esta actividad es acerca de hipertensión. Como se observa en la tabla, esta cuenta con variables como edad, sexo, nivel de colesterol, presión arterial, nivel de colesterol, etc. Así, utilizando la información buscaremos predecir si una persona tiene o no hipertensión.

import pandas as pd

df = pd.read_csv('hypertension_data.csv')

df.columns

print(df)

##Limpieza de Datos

#Como podemos ver, todas las columnas que tenemos son numéricas, por lo que no se deben utilizar dummies para pasar las variables categóricas a número binario.


df.columns

df.dtypes

#Con value_counts, vemos que la columna 'target' tiene valores muy similares en 0 1, es decir en si la persona tiene o no hipertensión, por lo que nuestra base de datos está balanceada.

df['target'].value_counts()

#Debido a que solamente tenemos 25 valores faltantes en la columna 'sexo', de 26058, se toma la decisión de eliminar las filas con valores faltantes. De igual manera, se eliminan los duplicados que tenemos.

df.shape

df.isnull().sum()

df = df.dropna()

df = df.drop_duplicates()

#Preparación para utilizar el algoritmo

#Definimos nuestra X y y, tomando en cuenta que el valor a predecir es 'target', tomando en cuenta el resto de las variables
X = df.drop('target', axis=1).values
y = df['target'].values

#Estandarización

#Se estandarizan los datos con el fin de tener su media en 0 y la desviación estándar en 1, lo cual ayuda con la distribución normal para los algoritmos
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

#Método SVM

#El algoritmo que se usará es Support Vector Machine, ya que se utiliza para clasificar

import numpy as np
from sklearn.model_selection import train_test_split

def svm_classify(X_train, y_train, X_test, C=10):
  """
  La función para la clasificación con SVM, utiliza los siguientes parámetros:

  Parámetros:
    x_train: Conjunto de datos de entrenamiento
    y_train: Etiquetas datos de entrenamiento
    x_test: Conjunto de datos de prueba
    C: Parámetro de regularización (su valor por defecto es 10)

  """

  #Inicio de los parámetros del modelo
  n_samples, n_features = X_train.shape
  w = np.zeros(n_features) #Vector de pesos
  b = 0 #sesgo

  #Entrenamiento del modelo
  for i in range(n_samples): #Para cada muestra calculamos el margen y actualizamos los pesos y el sesgo
    xi = X_train[i, :]
    yi = y_train[i]

    #Calculo de margen
    margin = 1 - yi * np.dot(w, xi) - b

    #Actualización de  pesos y el sesgo si el punto i está en el margen
    if margin > 0:
      w = w + C * yi * xi
      b = b + C * yi

  #Se realizan predicciones calcula el resultado antes de aplicar la función de signo
  y_pred = np.sign(np.dot(X_test, w) + b)#Dependiendo del signo se asigna 1 o -1

  return y_pred


#Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Usamos la función para entrenar el modelo
y_pred = svm_classify(X_train, y_train, X_test)

#Obtenemos la precisión
accuracy = np.mean(y_pred == y_test)

print("Precisión:", accuracy)