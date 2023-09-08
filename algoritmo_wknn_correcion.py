#Importación de librerías
import pandas as pd #Esta librería sirve para el manejo de bases de datos, en este caso se importa para leer el archivo de .csv que se utilizará
from sklearn.model_selection import train_test_split #Esta librería es para dividir arreglos en dos partes, entrenamiento y prueba
from sklearn.preprocessing import StandardScaler #Esta librería se usa para normalización, dónde la media será 0 y la desviación estándar 1
import numpy as np #La librería numpy se usa aquí para crear un arreglo 
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score #De sklearn.metrics se importan distintas métricas que ayudan a medir el desempeño de la clasiicación
import matplotlib.pyplot as plt #Esta librería se usa para poder realizar las gráficas, en este caso de barras, de cada métrica utilizada, para que se pueda hacer una mejor comparación

#Función de distancia ponderada
def distancia_ponderada(in1, in2, pesos):
    '''
    Usando los pesos que se asignan a las características, se calcula la distancia ponderada entre 2 puntos, en este caso, entre dos pacientes con sus respectivas características
    #lo cual se usa en WKNN, para el entrenamiento y porque con base a qué la similitud de los pacientes, se realiza la predicción
    '''

    #En el calculo de la distancia, primero se calcula la diferencia al cuadrado entre dos pacientes, luego, se toma en cuenta el peso que se asignó con base a la varianza de cada 
    #característica (variable) y, finalmente con zip, se combinan las características de ambos pacientes (in1 e in2), obteniendo el valor de la característcia del primer paciente, el 
    #valor del segundo paciente y el peso de la característica. Finalmente con [], se calcula la diferencia al cuadrado entre los dos puntos y, se multiplica por su peso correspondiente
    dif_cuadrada = [(a - b) ** 2 * peso for a, b, peso in zip(in1, in2, pesos)] 

    #El calculo que se devuelve la distancia ponderda entre los dos puntos, que se calcula con la raíz cuadrada de la suma de las diferencias cuadradas 
    return sum(dif_cuadrada) ** 0.5 

# Función de predicción
def prediccion(X_train, y_train, pesos, k, instancia):
    '''
    Con los datos de entrenamiento, los pesos definidos, el número de vecino, y la instancia, en esta función se realiza una predicción de clase de una instancia de prueba
    '''
    #Se crea una lista para almacenar las distancias
    distancias = []

    #En este ciclo vamos iterando sobre las instancias (pacientes/puntos) de entrenamiento
    for i, in_entrenada in enumerate(X_train):

        #Usamos la función de distancia ponderada (definida en la parte superior) y, le damos una instancia de prueba, una de entrenamiento y los pesos de las características
        dist = distancia_ponderada(in_entrenada, instancia, pesos)

        #Con .append se agrega a la lista de distancias la distancia ponderada calculada y la etiqueta correspondiente, la cual accedemos con iloc 
        distancias.append((dist, y_train.iloc[i]))

    #Se ordenan las distancias con .sort de menor a mayor, para poder identificar más fácilmente las menores distancias
    distancias.sort(key=lambda x: x[0])
    
    #Aquí, con base a las distancias calculadas, seleccionamos los vecinos más cercanos
    vecinos = distancias[:k]

    #Se crea un diccionario, el cual va a ayudar a contar las clases de los vecinos 
    clases = {0: 0, 1: 0}
    
    #Con este ciclo, se van contando cuántos vecinos hay en cada clase
    for dist, label in vecinos:
        clases[label] += 1
    
    #Con base a cuántos vecinos hay en cada clase, elegimos la clase más frecuente (clase predecida) y, eso es lo que nos devuelve nuestra función de predicción
    clase_predecida = max(clases, key=clases.get)
    return clase_predecida

#Usando la librería pandas leemos nuestro archivo, el cual es de datos de pacientes que presentan hipertensión
df = pd.read_csv('hypertension_data.csv')

#Dado que nuestra base de datos es muy extensa, de manera aleatoria, seleccionamos 4000 filas, con el fin de que nuestro modelo sea más rápido
df = df.sample(n=4000, random_state=42)

#Separar características y etiquetas

#La columna target, que es lo que se quiere predecir, por lo que la retiramos de nuestra X
X = df.drop('target', axis=1)

#Seleccionamos la columna de target y la guardamos en la y, para guardar los resultados de cada paciente
y = df['target']

#Se calcula la varianza de cada variable de nuestros predictores, para definir el peso de cada característica en el modelo 
X_train_var = X.var()

#Dividimos los datos en entrenamiento (80%) y prueba (20%), usando train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalización de datos

#Definimos StandarScaler para normalizar nuestros datos, de esta manera la media está 0 y la desviación estándar en 1, lo cual ayuda con la distribución normal para nuestro algoritmo
scaler = StandardScaler()

#Usamos el StandardScaler para normalizar nuestros datos de entrenamiento
X_train = scaler.fit_transform(X_train)

#Usamos el StandardScaler para normalizar nuestros datos de prueba
X_test = scaler.transform(X_test)

#Transformamos los valores de varianza de cada característica a un arreglo unidimensional y, usamos el StandardScaler para normalizar los datos
X_train_var = scaler.transform(np.array(X_train_var).reshape(1, -1))[0]

#Con base a la varianza que se presenta en cada variable, definimos el peso que tiene cada una de estas
pesos = X_train_var

#Se crea una lista vacía del número de vecinos, con el fin de almacenarlos para poder realizar gráficas posteriormente
valores_k = []

#Se crea una lista vacía de los valores de precisión, con el fin de almacenarlos para poder realizar gráficas posteriormente
resultados_precision = []

#Se crea una lista vacía de los valores de recall, con el fin de almacenarlos para poder realizar gráficas posteriormente
resultados_recall = []

#Se crea una lista vacía de los valores de f1, con el fin de almacenarlos para poder realizar gráficas posteriormente
resultados_f1 = []

#Se crea una lista vacía de los valores de accuracy, con el fin de almacenarlos para poder realizar gráficas posteriormente
resultados_accuracy = []

#Se crea un ciclo, en dónde vamos definiendo el número de vecinos, partiendo del tres y, subiendo de 4 en 4 hasta llegar a 27, con el fin de que nunca se tenga un número par de vecinos, provocando empates en la categoría
for k in range(3, 28, 4):

    #Se crea una lista vacía dónde almacenaremos las predicciones
    predicciones = []

    #Se añade a la lista de número de vecinos el vecino correspondiente, en este caso ser.a (3,7,11,15,19,23,27)
    valores_k.append(k)

    #Se genera un ciclo que recorra los datos de prueba
    for in_prueba in X_test:

        #Con la función prediccion, se realiza la prediccion de clase con cada fila de prueba
        predicted_class = prediccion(X_train, y_train, pesos, k, in_prueba)

        #Añadimos el valor a nuestra lista de predicciones
        predicciones.append(predicted_class)

    #Usando la métrica importada de precision_score, se calcula la precisión de cada escenario con k vecinos
    precision = precision_score(y_test, predicciones)
    print(f'Con {k} vecino(s), la precisión es de: {precision:.6f}')
    #Se guarda cada precisión en la lista para graficar posteriormente
    resultados_precision.append(precision)

    #Usando la métrica importada de recall, se calcula la precisión de cada escenario con k vecinos
    recall = recall_score(y_test, predicciones)
    print(f'Con {k} vecino(s), el recall es de: {recall:.6f}')
    #Se guarda cada recall en la lista para graficar posteriormente
    resultados_recall.append(recall)

    #Usando la métrica importada de f1, se calcula la precisión de cada escenario con k vecinos
    f1 = f1_score(y_test, predicciones)
    print(f'Con {k} vecino(s), el f1 score es de: {f1:.6f}')
    #Se guarda cada f1 en la lista para graficar posteriormente
    resultados_f1.append(f1)

    #Usando la métrica importada de accuracy, se calcula la precisión de cada escenario con k vecinos
    accuracy = accuracy_score(y_test, predicciones)
    print(f'Con {k} vecino(s), la accuracy es de: {accuracy:.6f}')
    #Se guarda cada accuracy en la lista para graficar posteriormente
    resultados_accuracy.append(accuracy)

#Usando plt, se crea el gráfico de barras para la precisión, de color naranja
plt.figure() #Se inicializa la figura
plt.bar(valores_k, resultados_precision, color='orange') #En el eje x tenemos el número de vecinos y en el eje y la precisión correspondiente
plt.xlabel('Valor de k') #El título de el eje x es 'Valor de k'
plt.ylabel('Puntaje Precisión') #El título de el eje x es 'Puntaje Precisión'
plt.title('Puntaje Precisión vs. Valor de k') #Se define el título de la gráfica
plt.ylim(0.6, 0.65)  #Establecer los límites del eje Y
plt.show() #El comando .show permite que se despliegue la figura en una nueva ventana

#Usando plt, se crea el gráfico de barras para recall, de color verde
plt.figure()#Se inicializa la figura
plt.bar(valores_k, resultados_recall, color='green') #En el eje x tenemos el número de vecinos y en el eje y el recall correspondiente
plt.xlabel('Valor de k') #El título de el eje x es 'Valor de k'
plt.ylabel('Puntaje Recall') #El título de el eje x es 'Puntaje Recall'
plt.title('Puntaje Recall vs. Valor de k') #Se define el título de la gráfica
plt.ylim(0.6, 0.8)  #Establecer los límites del eje Y
plt.show() #El comando .show permite que se despliegue la figura en una nueva ventana

#Usando plt, se crea el gráfico de barras para f1, de color azul
plt.figure() #Se inicializa la figura
plt.bar(valores_k, resultados_f1, color='blue') #En el eje x tenemos el número de vecinos y en el eje y la f1 correspondiente
plt.xlabel('Valor de k') #El título de el eje x es 'Valor de k'
plt.ylabel('Puntaje F1') #El título de el eje x es 'Puntaje F1'
plt.title('Puntaje F1 vs. Valor de k') #Se define el título de la gráfica
plt.ylim(0.6, 0.73)  #Establecer los límites del eje Y
plt.show() #El comando .show permite que se despliegue la figura en una nueva ventana

#Usando plt, se crea el gráfico de barras para accuracy, de color rojo
plt.figure() #Se inicializa la figura
plt.bar(valores_k, resultados_accuracy, color='red') #En el eje x tenemos el número de vecinos y en el eje y la accuracy correspondiente
plt.xlabel('Valor de k') #El título de el eje x es 'Valor de k'
plt.ylabel('Puntaje Accuracy') #El título de el eje x es 'Puntaje Accuracy'
plt.title('Puntaje Accuracy vs. Valor de k') #Se define el título de la gráfica
plt.ylim(0.6, 0.65)  # Establecer los límites del eje Y
plt.show() #El comando .show permite que se despliegue la figura en una nueva ventana

'''
Como podemos ver en los resultados de nuestro modelo WKNN, las métricas por cada número de vecinos son muy parecidas:

- En cuanto a los valores de precisión, la precisión más alta 0.630798 con 19 vecinos y, la más baja 0.620332 con 3 vecinos
- Con respecto al valor de recall, el más alto fue de 0.790698 con 19 vecinos y, el más bajo con 0.695349 con 3 vecinos
- El f1 score nos dio como máximo 0.701754 con 19 vecinos y, mínimo 0.655702
- Finalmente, la accuracy dio con el valor más alto 0.638750 con 19 vecinos y, más bajo 0.607500

Tomando en cuenta los resultados, podemos ver que el valor más bajo de los vecinos (3 vecinos) fue el que presentó peores resultados. 
Mientras que, 19 vecinos dió los mejores resultados en las 4 métricas. Esto a pesar de que no es la mayor cantidad de vecinos utilizada en este modelo. 
Esto se debe a que 19 vecinos ha resultado ser el mejor punto en el que se evita sufiente ruido para obtener los mejores resultados
'''
