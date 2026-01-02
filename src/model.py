import torch
import torch.nn as nn  # Importa los bloques de construcción (capas)
import torch.nn.functional as F  # Importa funciones operativas (activaciones)

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__() # Inicializa la clase base de PyTorch
        
        # Primer nivel de abstracción ---
        # Conv2d: Entrada 3 canales (RGB) -> Salida 16 mapas de características
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # Normaliza datos para acelerar aprendizaje
        
        # Segundo nivel ---
        # Entrada 16 (del anterior) -> Salida 32 nuevas características
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) # Estabiliza los 32 canales
        
        # Tercer nivel (detalles profundos) ---
        # Entrada 32 -> Salida 64 características complejas
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) # Estabiliza los 64 canales
    
        # CLASIFICADOR (Cerebro final) ---
        # Calcula el tamaño entrada: 64 canales * 3x3 pixeles (tamaño tras reducciones)
        self.fc1 = nn.Linear(64 * 3 * 3, 128) 
        self.fc2 = nn.Linear(128, num_classes) # Capa final: emite el diagnóstico

    def forward(self, x):
        # 1. Pasa por Conv1 -> Normaliza -> Activa (ReLU) -> Reduce tamaño a la mitad (Pool)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        
        # 2. Repite proceso (Conv2): Aprende más y vuelve a reducir tamaño
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        
        # 3. Repite proceso (Conv3): Extrae lo más complejo y reduce al mínimo (3x3)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        
        # 4. Aplanar: Convierte el cubo de datos 3D en una fila plana 1D
        x = x.view(x.size(0), -1) 
        
        # 5. Pensamiento: Procesa la info aplanada en la capa densa
        x = F.relu(self.fc1(x))
        
        # 6. Respuesta: Entrega los valores finales para cada clase
        x = self.fc2(x)
        return x