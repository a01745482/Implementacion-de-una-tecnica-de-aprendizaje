#Imporación de librerías
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Lectura de archivo
df = pd.read_csv('hypertension_data.csv')
df = pd.concat([df.head(2000),df.tail(2000)]) #Tomamos las primeras y últimas 200 filas del df para que el código corra más rápido, ya que es extenso (26083 filas)
# Separar características y etiquetas
X = df.drop('target', axis=1) #La columna target, que es lo que se quiere predecir
y = df['target'] #El resto de las características a tomar en cuenta para la predicción

#División de datos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalización
#De esta manera la media está 0 y la desviación estándar en 1, lo cual ayuda con la distribución normal para nuestro algoritmo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for k in range(1,10,2):
    #Calculo de distancia ponderada entre 2 instancias, usando los pesos que se asignan a las características
    def distancia_ponderada(in1, in2, pesos): 
        dif_cuadrada = (in1 - in2)**2
        dif_cuadrada_peso = dif_cuadrada * pesos
        return np.sqrt(np.sum(dif_cuadrada_peso))

    #Prediccion de la clase de una instancia
    def prediccion(X_train, y_train, pesos, k, instancia):
        distancias = []
        for i, in_entrenada in enumerate(X_train):
            dist = distancia_ponderada(in_entrenada, instancia, pesos)#Usando la distancia ponderada obtenemos la distancia entre las instancia de prueba y una instancia de entrenamient
            distancias.append((dist, y_train.iloc[i]))  # iloc para acceder a las etiquetas, guardamos las instancias en 'distancias'
        distancias.sort(key=lambda x: x[0])
        
        vecinos = distancias[:k]
        clases = {0: 0, 1: 0}
        
        for dist, label in vecinos:
            clases[label] += 1
        
        clase_predecida = max(clases, key=clases.get)
        return clase_predecida #Se devuelve la clase mayoritaria

    #Parámetros, definimos los peso basándonos en el número de columnas, en este caso 13
    pesos = np.ones(13)*0.5

    #Generamos las predicciones con el ciclo de las instancias para hacer la predicción de la clase por cada instancia, con los datos de entrenamiento
    predicciones = []
    for in_prueba in X_test:
        predicted_class = prediccion(X_train, y_train, pesos, k, in_prueba)
        predicciones.append(predicted_class) #Las guardamos

    # Evaluación del modelo
    accuracy = np.mean(predicciones == y_test)
    print(f'Con {k} vecino(s), la precisión es de: {accuracy:.2f}')

'''
Con base a los resultados, podemos notar que, la precisión aumenta conforme la cantidad de vecinos que se indican. Mientras menor sea la cantidad de 
vecinos, se obtendrá una mejor precisión. Esto se debe a que estamos considerando más instancias cercanas, por lo que tenemos más ruido,
nuestra discriminación disminuye, junto con la precisión
'''