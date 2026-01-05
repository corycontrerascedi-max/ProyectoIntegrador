import medmnist
import os

def download_datasets():
    print("--- INICIANDO DESCARGA DE DATOS (CROSSPATH AI) ---")
    
    # 1. Asegurar que exista la carpeta
    root = './assets/datasets'
    if not os.path.exists(root):
        os.makedirs(root)
        print(f"Carpeta creada: {root}")
    
    # 2. Descargar FUENTE: PathMNIST (Colon) - ESTE NO CAMBIA
    print("\n [1/2] Descargando datos de Colon (PathMNIST)...")
    medmnist.PathMNIST(split='train', download=True, root=root)
    print("Datos de Colon listos.")
    
    # 3. Descargar DESTINO: DermaMNIST (Piel) - ¡CAMBIO AQUÍ!
    # Antes era BreastMNIST. Ahora bajamos las 10,015 imágenes de piel.
    print("\n [2/2] Descargando datos de Piel (DermaMNIST)...")
    medmnist.DermaMNIST(split='train', download=True, root=root)
    print("Datos de Piel listos.")

    print("\n✨ ¡Todo listo! Tienes Colon (Fuente) y Piel (Destino).")

if __name__ == '__main__':
    download_datasets()