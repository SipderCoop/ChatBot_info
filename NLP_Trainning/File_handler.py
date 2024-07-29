# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:05:24 2024

@author: DJIMENEZ

Este script sirve para auxiliar en la extraccion de texto de diferentes tipos de archivos

"""

# Importar las bibliotecas necesarias
import os
from docx import Document
import fitz



# Función para leer archivos de texto
def read_txt(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except IOError as e:
        print(f"Error al leer el archivo: {file_path}")

# Función para leer archivos Word
def read_docx(file_path):
    try:
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except IOError as e:
        print(f"Error al leer el archivo: {file_path}")


# Función para leer archivos PDF
def read_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except IOError as e:
        print(f"Error al leer el archivo: {file_path}")



# Funcion que integra las demas en una sola
def extract_text_from_file(file_path):
    if os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            text = read_txt(file_path)
        elif ext == '.docx':
            text = read_docx(file_path)
        elif ext == '.pdf':
            text = read_pdf(file_path)
        else:
            print(f'Tipo de archivo no valido: {os.path.basename(file_path)}')
            text=None
    else:
        print(f'El archivo especificado no existe {os.path.basename(file_path)}')
        text=None
    return text
