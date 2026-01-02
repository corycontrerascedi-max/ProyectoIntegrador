import torch
import torch.nn as nn
import torch.optim as optim  # Herramientas para ajustar los pesos (aprender)
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
from model import Net  # Importamos el "cerebro" que definimos en el otro archivo
import os

def train_source():
    print("--- Entrenando Modelo Base (Colon) ---")
    
    # 1. DETECCIÓN DE HARDWARE
    # Revisa si tienes tarjeta gráfica NVIDIA (cuda). Si sí, la usa para ir rápido.
    # Si no, usa el procesador (cpu), que es más lento pero funciona en cualquier compu.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # 2. PREPARACIÓN DE DATOS
    info = INFO['pathmnist'] # Carga la info específica del dataset de Colon
    
    # Definimos cómo transformar las imágenes antes de que la IA las vea
    data_transform = transforms.Compose([
        transforms.ToTensor(), # Convierte imagen (JPG) a matriz matemática (Tensor)
        transforms.Normalize(mean=[.5], std=[.5]) # Escala valores entre -1 y 1 (ayuda matemática)
    ])
    
    root = './assets/datasets'
    
    # Crea la carpeta si no existe para evitar errores
    if not os.path.exists(root):
        os.makedirs(root)
        
    print("Cargando dataset...")
    # Descarga PathMNIST automáticamente si no lo tienes
    train_dataset = medmnist.PathMNIST(split='train', transform=data_transform, download=True, root=root)
    
    # El DataLoader es el "cargador". Agarra 128 imágenes, las revuelve (shuffle) 
    # y se las da a la IA en paquetes para no llenar la memoria RAM.
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    # 3. INICIAR EL MODELO
    print("Iniciando modelo...")
    # Crea una instancia de tu red neuronal y la manda a la GPU/CPU
    model = Net(num_classes=len(info['label'])).to(device)
    
    criterion = nn.CrossEntropyLoss() # La regla para medir el error (Loss)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # El algoritmo que ajusta las neuronas

    # 4. BUCLE DE ENTRENAMIENTO (Aquí aprende)
    EPOCHS = 3 # Cuántas veces repasará el libro completo (todas las imágenes)
    
    for epoch in range(EPOCHS):
        model.train() # Activa el "modo aprendizaje" (habilita capas especiales)
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Bucle interno: Procesa lote por lote (batch)
        for inputs, targets in train_loader:
            # Mover datos a la tarjeta gráfica
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze().long() # Ajuste de formato de etiquetas
            
            # A. Limpiar memoria: Borra los cálculos del paso anterior
            optimizer.zero_grad()
            
            # B. Examen: La IA intenta adivinar qué es la imagen
            outputs = model(inputs)
            
            # C. Calificación: Compara la predicción con la realidad
            loss = criterion(outputs, targets)
            
            # D. Aprendizaje (Backpropagation): Calcula quién tuvo la culpa del error
            loss.backward()
            
            # E. Ajuste: Actualiza los pesos de las neuronas para mejorar
            optimizer.step()
            
            # Cálculos estadísticos (para ver el progreso en pantalla)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        # Al final de cada época, imprimimos el reporte
        acc = 100. * correct / total
        print(f"Epoca {epoch+1}/{EPOCHS} -> Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

    # 5. GUARDAR RESULTADOS
    if not os.path.exists('./assets/models'):
        os.makedirs('./assets/models')
    
    # Guardamos el "state_dict" (los conocimientos adquiridos), no el código entero
    torch.save(model.state_dict(), './assets/models/modelo_base_colon.pth')
    print("✅ Modelo guardado.")

if __name__ == '__main__':
    train_source()