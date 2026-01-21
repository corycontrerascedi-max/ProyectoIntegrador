import torch
# Importamos el modulo de redes neuronales de PyTorch
import torch.nn as nn
# Importamos las funciones funcionales (como las activaciones y pooling)
import torch.nn.functional as F

# Definicion de la clase de la red neuronal
# Heredamos de nn.Module, que es la clase base para todos los modelos en PyTorch
class Net(nn.Module):
    def __init__(self, num_classes):
        # Inicializamos la clase padre (nn.Module) para heredar sus metodos
        super(Net, self).__init__()
        
        # --- BLOQUE CONVOLUCIONAL 1 ---
        # Fuente: LeCun et al. (1998) - Uso de convoluciones para extraer caracteristicas visuales
        # Entrada: 3 canales (Imagen a color RGB)
        # Salida: 16 canales (Mapas de caracteristicas detectados)
        # Kernel: 3x3 (Tamaño del filtro que recorre la imagen)
        # Padding: 1 (Agrega borde de ceros para mantener el tamaño de la imagen)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # Normalizacion por Lotes (Batch Normalization)
        # Fuente: Ioffe & Szegedy (2015) - Ayuda a entrenar mas rapido y estabiliza la red
        self.bn1 = nn.BatchNorm2d(16)
        
        # --- BLOQUE CONVOLUCIONAL 2 ---
        # Aumentamos la profundidad de 16 a 32 canales para detectar formas mas complejas
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # --- BLOQUE CONVOLUCIONAL 3 ---
        # Aumentamos la profundidad de 32 a 64 canales (detalles finos de la lesion)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
    
        # --- CAPAS COMPLETAMENTE CONECTADAS (CLASIFICADOR) ---
        # Aplanado: Calculamos el tamaño total de los datos antes de entrar a las neuronas lineales
        # NOTA IMPORTANTE: Asumimos que la imagen de entrada es de 24x24 pixeles.
        # Despues de 3 reducciones (Pooling), 24 se divide entre 2 tres veces: 24 -> 12 -> 6 -> 3.
        # Por eso el tamaño espacial final es 3x3.
        input_size_fc = 64 * 3 * 3
        
        # Primera capa lineal (Densa)
        self.fc1 = nn.Linear(input_size_fc, 128)
        
        # Segunda capa lineal (Salida)
        # El numero de salidas es igual al numero de clases (ej. 2: Melanoma vs Benigno)
        self.fc2 = nn.Linear(128, num_classes)

    # Definicion del flujo de datos (Como pasa la imagen por la red)
    def forward(self, x):
        # --- PASO 1: Primer nivel de caracteristicas ---
        # Aplicamos la convolucion
        x = self.conv1(x)
        # Normalizamos los datos
        x = self.bn1(x)
        # Aplicamos funcion de activacion ReLU
        # Fuente: Nair & Hinton (2010) - Elimina valores negativos (linealidad rectificada)
        x = F.relu(x)
        # Reducimos el tamaño a la mitad (Max Pooling)
        # Toma el valor mas alto en una ventana de 2x2
        x = F.max_pool2d(x, 2)
        
        # --- PASO 2: Segundo nivel de caracteristicas ---
        x = self.conv2(x)      # Convolucion
        x = self.bn2(x)        # Normalizacion
        x = F.relu(x)          # Activacion
        x = F.max_pool2d(x, 2) # Reduccion (Pooling)
        
        # --- PASO 3: Tercer nivel de caracteristicas ---
        x = self.conv3(x)      # Convolucion
        x = self.bn3(x)        # Normalizacion
        x = F.relu(x)          # Activacion
        x = F.max_pool2d(x, 2) # Reduccion (Pooling)
        
        # --- PASO 4: Aplanado (Flatten) ---
        # Convertimos el cubo 3D de datos en un vector plano 1D para el clasificador
        # x.size(0) es el tamaño del lote (batch size)
        # -1 le dice a PyTorch que calcule automaticamente el resto de las dimensiones
        x = x.view(x.size(0), -1)
        
        # --- PASO 5: Clasificacion ---
        # Pasamos por la primera capa densa
        x = self.fc1(x)
        x = F.relu(x) # Activacion
        
        # Pasamos por la capa de salida final
        x = self.fc2(x)
        
        # Retornamos el resultado (Logits)
        return x