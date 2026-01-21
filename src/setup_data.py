import medmnist
# Importamos la libreria del sistema operativo para gestionar carpetas
import os

def download_datasets():
    # Imprimimos mensaje de inicio para saber que el script corrio
    print("Iniciando proceso de descarga de datos MedMNIST...")
    
    # 1. Configuracion de carpetas
    # Definimos la ruta donde se guardaran los archivos descargados
    root = './assets/datasets'
    
    # Verificamos si la carpeta ya existe en el sistema
    if not os.path.exists(root):
        # Si no existe, la creamos automaticamente (makedirs crea subcarpetas si es necesario)
        os.makedirs(root)
        print(f"Carpeta creada exitosamente en: {root}")
    
    # 2. Descargar DOMINIO FUENTE (Source Domain)
    # Usaremos PathMNIST (Histopatologia de Colon)
    # Justificacion: Sirve para que la red aprenda texturas complejas antes de ver piel
    print("Descargando PathMNIST (Fuente: Colon)...")
    
    # La funcion descarga los datos si no existen (download=True)
    # split='train' prepara la configuracion para entrenamiento
    medmnist.PathMNIST(split='train', download=True, root=root)
    print("Descarga de PathMNIST completada.")
    
    # 3. Descargar DOMINIO OBJETIVO (Target Domain)
    # Usaremos DermaMNIST (Dermatoscopia de Piel)
    # Este es el dataset critico que contiene las clases de melanoma
    print("Descargando DermaMNIST (Objetivo: Piel)...")
    
    # DermaMNIST se basa en el HAM10000 (Human Against Machine)
    medmnist.DermaMNIST(split='train', download=True, root=root)
    print("Descarga de DermaMNIST completada.")

    # Confirmacion final
    print("Todos los datasets han sido verificados y almacenados.")

# Bloque de ejecucion principal
# Esto evita que se ejecute la descarga si importamos este archivo en otro lado
if __name__ == '__main__':
    download_datasets()