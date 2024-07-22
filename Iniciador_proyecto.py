"""
Created on Mon Mar 25 01:22:09 2024

@author: DJIMENEZ
"""
# Librerias necesarias -------------------------------------------------------------------------
import subprocess
import importlib
import sys
import os

# Obtener la ruta del directorio del archivo de script actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Establece el directorio de trabajo de donde se encuentra el script
sys.path.append(script_dir)
os.chdir(script_dir)


# Funciones para ayudar en el inicio de un proyecto -------------------------------------------------------------------------

# Funcion para instalar la libreria dada
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Funcion para revisar e instalar la libreria dada
def check_installation(package):
    try:
        importlib.import_module(package)
        print(f"{package} está instalado.")
    except ImportError:
        print(f"{package} no está instalado. Instalándolo ahora...")
        install(package)

# Funcion para crear un ambiente virtual
def create_virtual_environment(venv_name='venv', directory=None):
    # Definir la ruta del ambiente virtual
    venv_path = venv_name if directory is None else os.path.join(directory, venv_name)
    
    if os.path.exists(venv_path):
        print(f"El entorno virtual '{venv_name}' ya existe.")
    else:
        try:
            print(f"Creando el entorno virtual '{venv_name}' en {venv_path}...")
            subprocess.check_call([sys.executable, '-m', 'venv', venv_path])
            print(f"El entorno virtual '{venv_name}' ha sido creado exitosamente.")
        except subprocess.CalledProcessError as e:
            print(f"Ha ocurrido un error al crear el entorno virtual. Detalles del error: {e}")


# Funcion para activar el entorno virtual
# Tener en cuenta que la configuracion de la politica de ejecucion de scripts debe  estar en RemoteSigned
# En caso contrario, debe de entrar a la consola de Windows (cmd) y ejecutar el comando:
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# Esto cambia la configuracion, pero solo para el usuario actual, lo cual no necesita permisos de admin
def activate_virtual_environment(venv_path='venv'):
    # Construye el comando para activar el entorno
    activate_command = f"{venv_path}\\Scripts\\activate.bat"
    
    # Verifica si el archivo de activación existe
    if os.path.exists(activate_command):
        # Ejecuta el comando en el shell
        os.system(f'cmd /k "{activate_command}"')
    else:
        print("El script de activación no existe. ¿Está seguro de que el entorno virtual está creado?")

# Funcion para crear un archivo de .gitignore con los patrones de archivos mas comunes
def create_default_gitignore(directory=None):
    # Definir la ruta al archivo .gitignore
    gitignore_path = '.gitignore' if directory is None else os.path.join(directory, '.gitignore')
    
    # Contenido predeterminado para el archivo .gitignore
    gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Caches and logs
.cache
*.log
*.tmp

# Virtual environment
venv/
*.venv
.env

# Operating system files
.DS_Store
Thumbs.db

# IDE and editor directories
.idea/
.vscode/
*.swp
*~

# Node.js dependencies
node_modules/

# Build artifacts
/dist/
/build/
/target/

# Ignore secrets and credentials
secrets.yaml
config/credentials.yml
"""

    # Intentar escribir el contenido en el archivo .gitignore
    try:
        with open(gitignore_path, "w") as file:
            file.write(gitignore_content)
        print(f"Archivo .gitignore creado con configuración predeterminada en {gitignore_path}.")
    except Exception as e:
        print(f"Error al crear el archivo .gitignore en {gitignore_path}: {e}")



# Funciones para ayudar a compartir un proyecto -------------------------------------------------------------------------

# Funcion para crear el archivo requirements con las librarias instaladas en el directorio
def create_requirements_file(directory=None):
    # Define la ruta del archivo requirements.txt
    file_path = 'requirements.txt' if directory is None else os.path.join(directory, 'requirements.txt')
    
    try:
        # Abrir el archivo en el modo de escritura
        with open(file_path, 'w') as f:
            # Llamar a pip freeze y dirigir la salida al archivo
            subprocess.check_call([sys.executable, '-m', 'pip', 'freeze'], stdout=f)
        print(f'Archivo requirements.txt creado en {file_path}')
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar pip freeze: {e}")
    except Exception as ex:
        print(f"Error al crear el archivo requirements.txt en {file_path}: {ex}")


# Funcion para instalar las librarias especificadas en el archivo requirements
def install_requirements(directory=None):
    file_path = 'requirements.txt' if directory is None else os.path.join(directory, 'requirements.txt')
    if os.path.exists(file_path):
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', file_path])
            print("Las dependencias se han instalado correctamente.")
        except subprocess.CalledProcessError as e:
            print("Ha ocurrido un error al instalar las dependencias. Detalles del error:", e)
    else:
        print("El archivo requirements.txt no se encuentra en el directorio especificado.")

# Llama a la función
#create_virtual_environment()
activate_virtual_environment()
#create_default_gitignore()
#create_requirements_file()
#install_requirements()