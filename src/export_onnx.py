import torch
import torch.nn as nn
import os
import onnx # Libreria estandar para el intercambio de redes neuronales

# Importamos la definicion de la arquitectura
from model import Net

# --- CONFIGURACION DE RUTAS ---
# Usamos rutas relativas para que funcione en cualquier computadora
base_dir = './assets'
model_dir = os.path.join(base_dir, 'models')

# Ruta de entrada (Modelo PyTorch .pth entrenado)
MODEL_PATH = os.path.join(model_dir, 'modelo_final_piel.pth')
# Ruta de salida (Modelo optimizado .onnx)
ONNX_PATH = os.path.join(model_dir, 'crosspath_ai.onnx')

def export_to_onnx():
    print("Iniciando proceso de exportacion a ONNX (Open Neural Network Exchange)...")
    
    # 1. RECONSTRUCCION DE LA ARQUITECTURA
    # Debemos recrear exactamente la misma estructura que tenia el modelo al momento de guardarse.
    # El modelo nacio con 9 clases (Colon) y se modifico a 7 (Piel).
    
    print("Inicializando esqueleto del modelo...")
    # Paso A: Instanciar modelo base
    model = Net(num_classes=9)
    
    # Paso B: Replicar la modificacion de la ultima capa (Transfer Learning)
    n_classes_piel = 7
    num_ftrs = model.fc2.in_features
    model.fc2 = nn.Linear(num_ftrs, n_classes_piel)
    
    # 2. CARGA DE PESOS (STATE DICT)
    print(f"Cargando pesos desde: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("ERROR CRITICO: No se encontro el archivo .pth. Verifique haber ejecutado train_target.py.")
        return

    try:
        # Cargamos los tensores de pesos en la arquitectura
        model.load_state_dict(torch.load(MODEL_PATH))
        
        # IMPORTANTE: Cambiar a modo evaluacion
        # Esto desactiva capas aleatorias como Dropout y fija las estadisticas de BatchNorm.
        # Si no se hace, el modelo ONNX dara resultados erraticos.
        model.eval()
        print("Pesos cargados y modelo fijado en modo EVAL.")
        
    except Exception as e:
        print(f"Error cargando el modelo: {str(e)}")
        return

    # 3. TRAZADO DEL GRAFO (TRACING)
    # ONNX funciona "viendo" pasar un dato a traves de la red para mapear las operaciones.
    # Creamos un tensor aleatorio con las dimensiones esperadas: [Batch, Canales, Alto, Ancho]
    # MedMNIST usa imagenes de 28x28 pixeles.
    dummy_input = torch.randn(1, 3, 28, 28)

    # 4. EXPORTACION
    print(f"Generando archivo optimizado en: {ONNX_PATH}")
    
    torch.onnx.export(
        model,                  # El modelo de PyTorch
        dummy_input,            # El dato de ejemplo (Dummy Input)
        ONNX_PATH,              # Ruta de guardado
        export_params=True,     # Almacenar los pesos entrenados dentro del archivo
        opset_version=11,       # Version del set de operaciones (11 es muy estable para Raspberry)
        do_constant_folding=True, # Optimizacion: Pre-calcula operaciones constantes para acelerar inferencia
        input_names=['input'],  # Nombre del nodo de entrada (util para debug)
        output_names=['output'], # Nombre del nodo de salida
        dynamic_axes={          # Permitir tamaño de lote variable (util para procesar varias fotos a la vez)
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )

    print("\n--- EXPORTACION EXITOSA ---")
    print(f"Archivo generado: {ONNX_PATH}")
    print("Instrucciones: Copie este archivo a la carpeta del proyecto en la Raspberry Pi.")

if __name__ == '__main__':
    # Crear carpeta si no existe
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    export_to_onnx()