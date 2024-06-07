import math
import operator
import matplotlib.pyplot as plt
from tkinter import simpledialog, messagebox
import tkinter as tk

# Función para cargar los datos de Iris desde un archivo .data
def cargar_datos_iris(filename):
    features = []
    labels = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():  # Ignorar líneas en blanco
                data = line.strip().split(',')
                features.append([float(x) for x in data[:-1]])  # Convertir las características a números
                labels.append(data[-1])  # Agregar la etiqueta
    return features, labels

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# Función para obtener las k etiquetas más cercanas a un nuevo dato
def obtener_vecinos(training_set, training_labels, new_data, k):
    distances = []
    for i in range(len(training_set)):
        dist = distancia_euclidiana(new_data, training_set[i])  # Calcular la distancia entre el nuevo dato y los datos existentes
        distances.append((training_labels[i], dist))  # Agregar la etiqueta y la distancia a la lista
    distances.sort(key=operator.itemgetter(1))  # Ordenar la lista por distancia
    if k > len(distances):  # Si k es mayor que el número de datos disponibles, reducir k
        k = len(distances)
    neighbors = [item[0] for item in distances[:k]]  # Obtener las k etiquetas más cercanas
    return neighbors

# Función para predecir la etiqueta de un nuevo dato usando KNN
def predecir_clase(neighbors):
    class_votes = {}
    for neighbor in neighbors:
        if neighbor in class_votes:
            class_votes[neighbor] += 1
        else:
            class_votes[neighbor] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]

# Función para solicitar al usuario el valor de x e y para el nuevo dato
def obtener_nuevos_datos():
    x = simpledialog.askfloat("Ingresar valor", "Ingrese el valor de x para el nuevo dato (los datos registrados van de 4 a 8):")
    if x is not None and x < 0:
        messagebox.showerror("Valor inválido", "El valor de x no puede ser negativo. El programa se cerrará.")
        exit()  # Cerrar el programa si x es negativo
    y = simpledialog.askfloat("Ingresar valor", "Ingrese el valor de y para el nuevo dato (los datos registrados van de 2 a 4.5):")
    if y is not None and y < 0:
        messagebox.showerror("Valor inválido", "El valor de y no puede ser negativo. El programa se cerrará.")
        exit()  # Cerrar el programa si y es negativo
    return [x, y]

# Función para solicitar al usuario el número de vecinos (k) que desea usar
def obtener_valor_k():
    while True:
        k = simpledialog.askinteger("Número de vecinos", "Ingrese el número de vecinos (debe ser impar y estar entre 5 y 15):")
        if k is None:
            return None
        if k % 2 == 1 and 5 <= k <= 15:  # Verificar si es un número impar entre 5 y 15
            return k
        else:
            messagebox.showwarning("Número inválido", "Por favor ingrese un número impar entre 5 y 15.")

# Función para visualizar los datos y el nuevo dato ingresado por el usuario
def visualizar_datos(features, labels, new_data=None, new_label=None):
    colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
    plotted = set()  # Conjunto para almacenar las especies ya representadas
    for i in range(len(features)):
        if labels[i] not in plotted:  # Verificar si la especie ya ha sido representada
            plt.scatter(features[i][0], features[i][1], color=colors[labels[i]], label=labels[i])
            plotted.add(labels[i])
        else:
            plt.scatter(features[i][0], features[i][1], color=colors[labels[i]])

    if new_data:
        plt.scatter(new_data[0], new_data[1], color='black', marker='x', label=new_label)
        circle = plt.Circle((new_data[0], new_data[1]), 0.25, color='purple', fill=False)
        plt.gca().add_patch(circle)  # Dibujar un círculo alrededor del nuevo dato
    plt.xlabel('Alto del Sepalo')
    plt.ylabel('Ancho del Sepalo')
    plt.legend()
    plt.title('Conjunto de datos Iris')
    plt.show()

# Función principal del programa
def ejecutar_programa():
    features, labels = cargar_datos_iris('iris.data')
    while True:
        new_data = obtener_nuevos_datos()  # Solicitar al usuario los valores de x e y para el nuevo dato
        if new_data is not None:
            k = obtener_valor_k()  # Solicitar al usuario el número de vecinos
            if k is not None:
                # Obtener los vecinos más cercanos
                neighbors = obtener_vecinos(features, labels, new_data, k)
                # Construir el mensaje con información de los vecinos
                message = "Vecinos más cercanos:\n"
                for neighbor in neighbors:
                    message += f"Clase: {neighbor}\n"
                
                # Mostrar el mensaje en una alerta
                messagebox.showinfo("Vecinos más cercanos", message)
                # Predecir la clase del nuevo dato
                predicted_label = predecir_clase(neighbors)
                # Llamar a la función para visualizar los datos con la etiqueta predicha
                visualizar_datos(features, labels, new_data, predicted_label)
        else:
            break  # Salir del bucle si no se ingresan nuevos datos

# Ejecutar el programa
root = tk.Tk()
root.withdraw()  # Ocultar la ventana principal de Tkinter
ejecutar_programa()
