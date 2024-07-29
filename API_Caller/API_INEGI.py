# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:20:28 2024

@author: DJIMENEZ

"""
# Librerias necesarias -------------------------------------------------------------------------

# Import the os module to interact with the operating system, like setting the working directory.
import os
# Import the sys module to manipulate the Python runtime environment, such as the system path.
import sys
# Importa pandas
import pandas as pd
# Import numpy
import numpy as np
# Import datetime
from datetime import datetime
# Import load_dotenv function from dotenv to load environment variables from a .env file.
from dotenv import load_dotenv
# Import requests for making HTTP requests to APIs.
import requests
# Import json for encoding and decoding JSON data.
import json


# Configuracion inicial -------------------------------------------------------------------------

# Set the current script's directory as the working directory to ensure relative paths work correctly.
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
os.chdir(script_dir)

# Load environment variables from a .env file located in the same directory.
load_dotenv()

# Tokens
Banxico_Token = os.environ.get("Banxico_Token")
INEGI_Token = os.environ.get("INEGI_Token")

# Funciones -------------------------------------------------------------------------

# Funcion para cambiar la presentacion de los periodos de tiempo de la serie 
# de acuerdo a las especificacionesde la metadata de la API de INEGI
def BIE_freq_handler(time_periods,frequency_id):

    # Obtener datos de API
    url = f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/CL_FREQ/{frequency_id}/es/BIE/2.0/{INEGI_Token}?type=json"
    response = requests.get(url)
    response.encoding = 'latin1'

    # Extraer y convertir datos 
    data_json = response.json()
    data_serie = data_json['CODE'][0]
    freq_str = data_serie['Description'].encode('latin1').decode('latin1')
    
    if frequency_id == 6:

        time_periods_formatted = []
        for period in time_periods:
            year, quarter = map(int, period.split('/'))
            time_periods_formatted.append(f"{quarter}T{year}")

    elif frequency_id == 8:
        
        time_periods_formatted = [pd.to_datetime(period).strftime('%Y%m') for period in time_periods]

    return freq_str, time_periods_formatted


def BIE_unit_handler(unit_id):
    # Obtener datos de API
    url = f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/CL_UNIT/{unit_id}/es/BIE/2.0/{INEGI_Token}?type=json"
    response = requests.get(url)
    response.encoding = 'utf-8'

    # Extraer y convertir datos 
    data_json = response.json()
    data_serie = data_json['CODE'][0]
    unit_str = data_serie['Description'].encode('').decode('latin1')

    return unit_str


# Function to get the GDP value
def get_BIE_data(indicator_id,last_data=False):
    # Cambia el formato para adecuarse a la API
    if last_data == True:
        last_data = 'true'
    else:
        last_data = 'false'
    
    # Obtener datos de API
    url = f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/{str(indicator_id)}/es/0700/{last_data}/BIE/2.0/{INEGI_Token}?type=json"
    response = requests.get(url)
    response.encoding = 'latin1'

    # Extraer y convertir datos 
    data_json = json.loads(response.text)
    data_serie = data_json['Series'][0]
    freq = int(data_serie['FREQ'])
    unit = int(data_serie['UNIT'])
    obs = data_serie['OBSERVATIONS']

    # Extrae OBS_VALUE y TIME_PERIOD de los datos
    obs_values = [float(entry['OBS_VALUE']) for entry in obs]
    time_periods = [entry['TIME_PERIOD'] for entry in obs]

    # Transforma los periodos y frecuencia para que sea mas legible
    freq_str, time_periods_formatted = BIE_freq_handler(time_periods,freq)

    # Tranforma la unidad de medida para que sea mas legible
    unit_str = BIE_unit_handler(unit)

    # Creating a Pandas series
    serie = pd.Series(obs_values, index=time_periods_formatted)
    serie = serie.to_dict()
    
    # Create json with the series and important metadata
    json_data = {"unidades":unit_str,"frecuencia":freq_str,"serie":serie}
    serie_json = json.dumps(json_data, indent=4)
    
    return serie_json



# Ejemplo de uso


if __name__ == '__main__':

    # IDs 736181, 454527, 628208
    serie = get_BIE_data(628208,True)
    print(serie)

