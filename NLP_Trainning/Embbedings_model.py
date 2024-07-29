# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:05:24 2024

@author: DJIMENEZ

Este script sirve para crear embeddings a partir de archivos para su uso en modelos de NLP

"""

# Importar las bibliotecas necesarias -------------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import create_optimizer
from datasets import Dataset

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

# Paso 2: Tokenizar y Crear Dataset
def create_dataset(data, labels, tokenizer):
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    
    encodings = tokenizer(data, truncation=True, padding=True)
    dataset = Dataset.from_dict({**encodings, 'labels': labels.tolist()})
    return dataset, mlb

# Paso 3: Entrenar el Clasificador Multietiqueta
def train_classifier(dataset,tokenizer, model_name='bert-base-uncased', num_labels=None, batch_size=8, epochs=3):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    tf_dataset = dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["labels"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    num_train_steps = len(tf_dataset) * epochs
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_train_steps=num_train_steps, num_warmup_steps=0)
    
    model.compile(optimizer=optimizer, loss=model.compute_loss)
    model.fit(tf_dataset, epochs=epochs)
    
    return model

# Paso 4: Predecir el Tema para una Pregunta del Usuario
def predict_topic(question, tokenizer, model, mlb):
    inputs = tokenizer(question, return_tensors="tf", truncation=True, padding=True)
    outputs = model(inputs)
    predictions = tf.nn.sigmoid(outputs.logits)
    predicted_labels = tf.where(predictions > 0.5, 1, 0)
    topics = mlb.inverse_transform(predicted_labels.numpy())
    return topics

# Función principal
def main():
    data_dir = 'path/to/your/data'  # Reemplazar con el directorio de tus datos
    data_dir = os.path.normpath(data_dir)
    data, labels = load_data(data_dir)  # Cargar los datos y las etiquetas

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset, mlb = create_dataset(data, labels, tokenizer)
    num_labels = len(mlb.classes_)
    
    model = train_classifier(dataset, model_name=model_name, num_labels=num_labels)

    while True:  # Bucle para recibir preguntas del usuario
        question = input("Enter your question: ")  # Pedir al usuario que ingrese una pregunta
        topics = predict_topic(question, tokenizer, model, mlb)  # Predecir el tema de la pregunta
        print(f"Predicted Topics: {topics}")  # Imprimir los temas predichos

if __name__ == "__main__":
    main()