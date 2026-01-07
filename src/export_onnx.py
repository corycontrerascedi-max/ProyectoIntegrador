import torch
import torch.nn as nn
import os
import onnx

# Importamos tu arquitectura
from model import Net

# --- RUTAS ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Ruta del modelo entrenado (Piel)
MODEL_PATH = os.path.join(base_dir, 'assets', 'models', 'modelo_final_piel.pth')
# Ruta donde guardaremos el archivo optimizado
ONNX_PATH = os.path.join(base_dir, 'assets', 'models', 'crosspath_ai.onnx')

def export_to_onnx():
    print("--- INICIANDO EXPORTACIÓN A ONNX ---")
    
    # 1. RECONSTRUIR LA ARQUITECTURA EXACTA
    # Recuerda: Tu modelo nació con 9 clases (Colon) y luego le operamos la capa final a 7 (Piel).
    # Debemos repetir esos pasos para que los pesos encajen.
    
    print("Reconstruyendo arquitectura del modelo...")
    # Paso A: Crear base original
    model = Net(num_classes=9) 
    
    # Paso B: Reemplazar la capa final por la de 7 clases (igual que en el entrenamiento)
    # NOTA: En DermaMNIST hay 7 clases.
    n_classes_piel = 7 
    num_ftrs = model.fc2.in_features
    model.fc2 = nn.Linear(num_ftrs, n_classes_piel)
    
    # 2. CARGAR LOS PESOS ENTRENADOS
    print(f"Cargando pesos desde: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("ERROR: No encuentro el modelo .pth. ¿Ya corriste la Fase 2?")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() # Poner en modo evaluación (apaga Dropout, congela Batchnorm)
        print("Pesos cargados correctamente.")
    except Exception as e:
        print(f"Error al cargar state_dict: {e}")
        return

    # 3. CREAR UN DATO "FANTASMA" (DUMMY INPUT)
    # ONNX necesita ver pasar una foto de prueba para entender el camino de las neuronas.
    # Formato: [Batch=1, Canales=3, Alto=28, Ancho=28] (Tamaño estándar MedMNIST)
    dummy_input = torch.randn(1, 3, 28, 28)

    # 4. EXPORTAR
    print(f"Exportando a: {ONNX_PATH}")
    torch.onnx.export(
        model,                      # El modelo
        dummy_input,                # El dato fantasma
        ONNX_PATH,                  # Dónde guardar
        export_params=True,         # Guardar los pesos entrenados dentro del archivo
        opset_version=11,           # Versión compatible con Raspberry Pi
        do_constant_folding=True,   # Optimización matemática (hace el modelo más rápido)
        input_names=['input'],      # Nombre de la entrada (lo usaremos en la App)
        output_names=['output'],    # Nombre de la salida (lo usaremos en la App)
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Permitir varios tamaños de batch
    )

    print("\n Tu modelo 'crosspath_ai.onnx' está listo.")
    print("   Este es el archivo que copiarás a la Raspberry Pi.")

if __name__ == '__main__':
    export_to_onnx()