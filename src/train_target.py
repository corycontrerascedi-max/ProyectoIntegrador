import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import os
import numpy as np

# Librerias para graficar resultados
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Importamos la arquitectura definida previamente
from model import Net

# --- FUNCIONES AUXILIARES DE VISUALIZACION ---

def plot_training_results(loss_list, acc_list):
    """
    Genera y guarda las curvas de aprendizaje (Loss y Accuracy).
    Ayuda a detectar Overfitting (si la Loss baja pero el Accuracy se estanca).
    """
    plt.figure(figsize=(12, 5))
    
    # Grafica 1: Perdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='Entrenamiento', color='red')
    plt.title('Curva de Convergencia (Loss)')
    plt.xlabel('Epocas')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Grafica 2: Precision (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(acc_list, label='Entrenamiento', color='blue')
    plt.title('Evolucion de la Precision')
    plt.xlabel('Epocas')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    # Guardar grafica
    plt.savefig('grafica_entrenamiento.png')
    print("Graficas guardadas en 'grafica_entrenamiento.png'.")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Genera un mapa de calor mostrando que clases se confunden entre si.
    Vital para diagnositco medico (ver falsos negativos).
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalizamos la matriz para ver porcentajes (opcional, aqui usamos absolutos)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prediccion del Modelo')
    plt.ylabel('Etiqueta Real (Ground Truth)')
    plt.title('Matriz de Confusion - DermaMNIST')
    
    plt.savefig('matriz_confusion.png')
    print("Matriz de confusion guardada en 'matriz_confusion.png'.")
    plt.close()

# --- FUNCION PRINCIPAL DE ENTRENAMIENTO ---

def train_target():
    print("Iniciando Fase 2: Transfer Learning (Dominio Objetivo: Piel)...")
    
    # 1. CONFIGURACION DE RUTAS Y HARDWARE
    base_dir = './assets'
    model_dir = os.path.join(base_dir, 'models')
    
    # Rutas de archivos
    source_path = os.path.join(model_dir, 'modelo_base_colon.pth')
    target_path = os.path.join(model_dir, 'modelo_final_piel.pth')
    
    # Verificacion de hardware (GPU vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo seleccionado: {device}")

    # Verificar si existe el modelo pre-entrenado
    if not os.path.exists(source_path):
        print("ERROR CRITICO: No se encontro el modelo base (Colon). Ejecute train_source.py primero.")
        return

    # 2. PREPARACION DE DATOS CON AUMENTO (DATA AUGMENTATION)
    # Fuente: Shorten & Khoshgoftaar (2019) - "A survey on Image Data Augmentation".
    # Justificacion: Al rotar y alterar las imagenes, evitamos que la red memorice posiciones fijas.
    
    print("Configurando Data Augmentation...")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),      # Espejo horizontal
        transforms.RandomVerticalFlip(),        # Espejo vertical
        transforms.RandomRotation(20),          # Rotacion +/- 20 grados
        transforms.ColorJitter(brightness=0.1), # Variacion leve de brillo (simula diferentes lamparas)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Para validacion/test NO aplicamos aumento, solo normalizacion
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Descarga y carga de datos
    print("Cargando dataset DermaMNIST...")
    train_dataset = medmnist.DermaMNIST(split='train', transform=train_transform, download=True, root=os.path.join(base_dir, 'datasets'))
    test_dataset = medmnist.DermaMNIST(split='test', transform=test_transform, download=True, root=os.path.join(base_dir, 'datasets'))
    
    # Definimos el tamaño del lote (Batch Size)
    BATCH_SIZE = 128
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. CARGA DEL MODELO Y TRANSFER LEARNING
    print("Cargando arquitectura y pesos previos...")
    
    # Instanciamos el modelo con 9 clases (Configuracion original de Colon)
    model = Net(num_classes=9)
    model.load_state_dict(torch.load(source_path, map_location=device))
    
    # TECNICA: Freezing (Congelamiento)
    # Congelamos los pesos de las capas convolucionales para no perder el aprendizaje de texturas
    for param in model.parameters():
        param.requires_grad = False
        
    # Reemplazo de la ultima capa (Head Replacement)
    # Obtenemos el numero de entradas de la ultima capa lineal
    num_features = model.fc2.in_features
    # DermaMNIST tiene 7 clases
    n_classes_piel = 7 
    
    # Sobrescribimos la capa final. Esta nueva capa SI tendra gradientes activos.
    model.fc2 = nn.Linear(num_features, n_classes_piel)
    
    # Enviamos modelo a GPU
    model.to(device)
    
    # Definimos optimizador solo para los parametros que requieren gradiente (la ultima capa)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # 4. MANEJO DE DESBALANCE DE CLASES (COST-SENSITIVE LEARNING)
    # Fuente: Johnson & Khoshgoftaar (2019) - "Survey on deep learning with class imbalance".
    # Problema: Hay muchos 'Nevus' y pocos 'Melanomas'. La red tiende a ignorar el melanoma.
    # Solucion: Penalizar mas fuerte cuando se equivoca en las clases minoritarias.
    
    print("Calculando pesos para balanceo de clases...")
    # Extraemos etiquetas para contar frecuencias
    try:
        targets = train_dataset.labels.squeeze()
    except:
        targets = [y for _, y in train_dataset]
        
    class_counts = np.bincount(targets)
    total_samples = sum(class_counts)
    
    # Formula de Peso Inverso: W = Total / (Num_Clases * Frecuencia)
    class_weights = total_samples / (n_classes_piel * class_counts)
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"Pesos calculados: {np.round(class_weights, 2)}")
    
    # Aplicamos los pesos a la funcion de perdida
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # 5. BUCLE DE ENTRENAMIENTO
    EPOCHS = 15
    loss_history = []
    acc_history = []
    
    print("Iniciando entrenamiento...")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets_batch in train_loader:
            inputs, targets_batch = inputs.to(device), targets_batch.to(device)
            targets_batch = targets_batch.squeeze().long()
            
            # Reset de gradientes
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets_batch)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metricas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets_batch.size(0)
            correct += predicted.eq(targets_batch).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        
        print(f"Epoca [{epoch+1}/{EPOCHS}] -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # 6. GUARDADO Y EVALUACION
    torch.save(model.state_dict(), target_path)
    print(f"Modelo final guardado en: {target_path}")
    
    # Generar graficas
    plot_training_results(loss_history, acc_history)
    
    # Generar matriz de confusion usando el set de prueba
    print("Generando matriz de confusion con datos de prueba...")
    all_preds = []
    all_targets = []
    
    model.eval() # Modo evaluacion (desactiva dropout/batchnorm dinamico)
    with torch.no_grad(): # Desactivamos calculo de gradientes para ahorrar memoria
        for inputs, targets_batch in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets_batch.squeeze().numpy())
            
    # Nombres de las clases en DermaMNIST
    class_names = ['Actinic', 'BCC', 'Benign', 'Dermato', 'Melanoma', 'Nevus', 'Vasc']
    plot_confusion_matrix(all_targets, all_preds, class_names)

if __name__ == '__main__':
    train_target()