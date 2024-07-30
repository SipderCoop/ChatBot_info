# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:05:24 2024

@author: DJIMENEZ

Este script sirve para crear embeddings a partir de archivos para su uso en modelos de NLP

"""

# Importar las bibliotecas necesarias -------------------------------------------------------------
import os
import pickle

from File_handler import extract_text_from_file


# Paso 1: Preparación de Datos
def load_data(data_dir):
    # Listas para almacenar los textos y sus etiquetas
    data = []  
    labels = []

    # Iterar sobre todos los directorios y archivos
    for root, dirs, files in os.walk(data_dir):  
        
        # Iterar sobre cada archivo en el directorio
        for file_name in files:  
            # Extraccion de texto de los archivos
            file_path = os.path.join(root, file_name)
            text = extract_text_from_file(file_path) 

            if text:
                # Generacion de etiquetas a partir de las carpetas y subcarpetas
                relative_root = os.path.relpath(root, data_dir)
                file_labels = relative_root.split(os.sep)

                # Se agrega a una tupla para que no se pueda modificar y por ultimo a una lista
                data.append(text)
                labels.append(file_labels)

    return data, labels

# Guardar datos en un archivo
def save_data(data, labels, filename='data.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump((data, labels), file)

# Cargar datos desde un archivo
def load_saved_data(filename='data.pkl'):
    with open(filename, 'rb') as file:
        data, labels = pickle.load(file)
    return data, labels

if __name__ == "__main__":
    data_dir = 'C:/Users/David Cooper/Desktop/Comisiones'  # Reemplazar con el directorio de tus datos
    data_dir = os.path.normpath(data_dir)
    data, labels = load_data(data_dir)  # Cargar los datos y las etiquetas
    data, labels = load_saved_data()  # Guardar los datos procesados

    print(f'numero de labels: {len(labels)}')
    print(f'numero de textos: {len(data)}')
        