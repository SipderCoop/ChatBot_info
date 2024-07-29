# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:05:24 2024

@author: DJIMENEZ

Este script funciona con la version 21.0.1 de la libreria python-telegram-bot

"""

# Librerias necesarias -------------------------------------------------------------------------

# Import the os module to interact with the operating system, like setting the working directory.
import os
# Import the sys module to manipulate the Python runtime environment, such as the system path.
import sys
# Import load_dotenv function from dotenv to load environment variables from a .env file.
from dotenv import load_dotenv
# Import logging to handle errors
import logging

# Importa librerias locales
import API_Caller.API_INEGI as API_INEGI


# Configuraciones iniciales -------------------------------------------------------------------------

# Set the current script's directory as the working directory to ensure relative paths work correctly.
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
os.chdir(script_dir)

# Funciones del Bot -------------------------------------------------------------------------

import spacy

nlp = spacy.load("es_core_news_sm")
# O para el modelo mediano: nlp = spacy.load("es_core_news_md")

doc = nlp("Este es un texto de ejemplo en español Marzo 2024.")
for token in doc:
    print(token.text, token.lemma_, token.pos_)
