import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import os

# Importamos la arquitectura de red neuronal definida en el paso anterior
from model import Net

def train_source():
    print("Iniciando entrenamiento del Modelo Base (Dominio Fuente: Colon)...")
    
    # 1. CONFIGURACION DE HARDWARE
    # Verificamos si existe disponibilidad de GPU (CUDA) para acelerar el calculo matricial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de procesamiento seleccionado: {device}")
    
    # 2. PREPARACION DE DATOS Y PRE-PROCESAMIENTO
    # Cargamos la informacion del dataset PathMNIST (Colon)
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    num_classes = len(info['label']) # Obtenemos el numero de clases (9 tipos de tejido)
    
    # Definimos la cadena de transformaciones (Pipeline)
    # Fuente: Shorten & Khoshgoftaar (2019) - La normalizacion acelera la convergencia del descenso de gradiente
    data_transform = transforms.Compose([
        transforms.ToTensor(), # Convierte imagen a Tensor de PyTorch (Float32)
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normaliza los valores de pixeles al rango [-1, 1]
    ])
    
    root = './assets/datasets'
    
    # Cargamos el dataset de entrenamiento
    print("Cargando dataset PathMNIST...")
    train_dataset = medmnist.PathMNIST(split='train', transform=data_transform, download=True, root=root)
    
    # Configuramos el DataLoader
    # Batch Size 128: Procesamos 128 imagenes simultaneamente para estimar el gradiente
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    # 3. INSTANCIACION DEL MODELO
    print(f"Inicializando red neuronal para {num_classes} clases...")
    model = Net(num_classes=num_classes)
    model.to(device) # Movemos el modelo a la memoria de la GPU/CPU
    
    # 4. DEFINICION DE HIPERPARAMETROS
    # Funcion de Perdida: CrossEntropyLoss
    # Fuente: Rubinstein (1999) - Estandar para clasificacion multiclase, penaliza logaritmicamente el error
    criterion = nn.CrossEntropyLoss()
    
    # Optimizador: Adam (Adaptive Moment Estimation)
    # Fuente: Kingma & Ba (2014) - Ajusta la tasa de aprendizaje automaticamente por parametro
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. BUCLE DE ENTRENAMIENTO (TRAINING LOOP)
    EPOCHS = 3 # Numero de veces que el modelo vera todo el dataset
    
    for epoch in range(EPOCHS):
        model.train() # Ponemos el modelo en modo entrenamiento (habilita BatchNorm y Dropout)
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Iteramos sobre los lotes de datos
        for inputs, targets in train_loader:
            # Enviamos datos al dispositivo
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Las etiquetas en MedMNIST vienen como [Batch, 1], necesitamos [Batch]
            targets = targets.squeeze().long()
            
            # PASO A: Reiniciar gradientes
            # Borramos los gradientes calculados en la iteracion anterior
            optimizer.zero_grad()
            
            # PASO B: Inferencia (Forward Pass)
            # Calculamos la prediccion del modelo
            outputs = model(inputs)
            
            # PASO C: Calculo de error (Loss)
            loss = criterion(outputs, targets)
            
            # PASO D: Retropropagacion (Backpropagation)
            # Calculamos la derivada del error respecto a cada peso
            # Fuente: Rumelhart et al. (1986)
            loss.backward()
            
            # PASO E: Optimizacion
            # Actualizamos los pesos restando el gradiente multiplicado por el learning rate
            optimizer.step()
            
            # Calculo de metricas para monitoreo
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        # Reporte de progreso al final de la epoca
        acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoca [{epoch+1}/{EPOCHS}] -> Perdida: {avg_loss:.4f} | Precision: {acc:.2f}%")

    # 6. EXPORTACION DEL MODELO
    output_dir = './assets/models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_path = os.path.join(output_dir, 'modelo_base_colon.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Modelo entrenado guardado exitosamente en: {save_path}")

if __name__ == '__main__':
    train_source()