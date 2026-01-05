import medmnist
import os

def download_datasets():
    print("--- INICIANDO DESCARGA DE DATOS ---")
    
    # 1. Asegurar que exista la carpeta
    root = './assets/datasets'
    if not os.path.exists(root):
        os.makedirs(root)
        print(f"Carpeta creada: {root}")
    
    # 2. Descargar PathMNIST (Colon - Fase 1)
    print("\n Descargando datos de Colon (PathMNIST)...")
    medmnist.PathMNIST(split='train', download=True, root=root)
    print("Datos de Colon listos.")
    
    # 3. Descargar BreastMNIST (Mama - Fase 2)
    # Lo bajamos de una vez para que ya lo tengan listo
    print("\n Descargando datos de Mama (BreastMNIST)...")
    medmnist.BreastMNIST(split='train', download=True, root=root)
    print("Datos de Mama listos.")

    print("\n Todo listo! Los datos están en 'assets/datasets'.")

if __name__ == '__main__':
    download_datasets()