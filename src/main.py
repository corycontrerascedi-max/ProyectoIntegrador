import torch

def verificar_entorno():
    print("Verificación de Hardware para IA")
    
    # Comprobar si PyTorch detecta la GPU
    if torch.cuda.is_available():
        nombre_gpu = torch.cuda.get_device_name(0)
        print(f"GPU detectada: {nombre_gpu}")
        print(f"Versión de CUDA activa: {torch.version.cuda}")
    else:
        print("No se detectó GPU")
        print("Sugerencia: Revisa que tus drivers de NVIDIA estén actualizados.")

if __name__ == "__main__":
    verificar_entorno()